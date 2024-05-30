using System.Runtime.InteropServices;
using System.IO.MemoryMappedFiles;
using System.Text;
using System.Numerics.Tensors;
using System.Diagnostics;
using System.Globalization;
using CommandLine;

Parser.Default.ParseArguments<Options>(args)
    .WithParsed(Run)
    .WithNotParsed(errs => { foreach (var error in errs) Console.Error.WriteLine(error); Environment.Exit(1); });
return;

void Run(Options opts) {
    if (!File.Exists(opts.TokenizerPath)) { Console.Error.WriteLine($"couldn't load tokenizer {opts.TokenizerPath}"); Environment.Exit(1); }
    if (!File.Exists(opts.CheckpointPath)) { Console.Error.WriteLine($"couldn't load model {opts.CheckpointPath}"); Environment.Exit(1); }

    // parameter validation/overrides
    if (opts.Seed <= 0) opts.Seed = (ulong)DateTimeOffset.Now.ToUnixTimeSeconds();
    if (opts.Temperature < 0.0) opts.Temperature = 0.0f;
    if (opts.Topp < 0.0 || 1.0 < opts.Topp) opts.Topp = 0.9f;
    if (opts.Steps < 0) opts.Steps = 0;
    if (opts.Verbose) Log("program started.");

    // build the Transformer via the model .bin file
    var transformer = BuildTransformer(opts);
    if (opts.Steps == 0 || opts.Steps > transformer.config.seq_len) opts.Steps = transformer.config.seq_len; // override to ~max length
    if (opts.Verbose) Log("transformer created.");

    // build the Tokenizer via the tokenizer .bin file
    var tokenizer = BuildTokenizer(transformer.config.vocab_size, opts);
    if (opts.Verbose) Log("tokenizer created.");

    // build the Sampler
    var sampler = BuildSampler(transformer.config.vocab_size, opts);

    switch (opts.Mode) {
        case Mode.Generate:
            if (opts.Verbose) Log("generation started.");
            Generate(ref transformer, ref tokenizer, ref sampler, opts);
            break;
        case Mode.Chat:
            if (opts.Verbose) Log("chat started.");
            Chat(ref transformer, ref tokenizer, ref sampler, opts);
            break;
        default:
            throw new ArgumentException("unknown mode");
    }
    if (opts.Verbose) Log("finished successfully.");
}

static RunState BuildRunState(Config p) {
    int kvDim = p.dim * p.n_kv_heads / p.n_heads;
    return new RunState {
        x = new float[p.dim],
        xb = new float[p.dim],
        xb2 = new float[p.dim],
        hb = new float[p.hidden_dim],
        hb2 = new float[p.hidden_dim],
        q = new float[p.dim],
        k = new float[p.dim],
        v = new float[p.dim],
        key_cache = new float[p.n_layers * p.seq_len * kvDim],
        value_cache = new float[p.n_layers * p.seq_len * kvDim],
        att = new float[p.n_heads * p.seq_len],
        logits = new float[p.vocab_size]
    };
}

static void MemoryMapWeights(ref TransformerWeights w, ref Config p, UnmanagedMemoryAccessor a, bool sharedWeights) {
    long position = 0;

    w.token_embedding_table = GetSpan(p.vocab_size * p.dim);
    w.rms_att_weight = GetSpan(p.n_layers * p.dim);
    w.wq = GetSpan(p.n_layers * p.dim * p.dim);
    w.wk = GetSpan(p.n_layers * p.dim * p.dim);
    w.wv = GetSpan(p.n_layers * p.dim * p.dim);
    w.wo = GetSpan(p.n_layers * p.dim * p.dim);
    w.rms_ffn_weight = GetSpan(p.n_layers * p.dim);
    w.w1 = GetSpan(p.n_layers * p.dim * p.hidden_dim);
    w.w2 = GetSpan(p.n_layers * p.hidden_dim * p.dim);
    w.w3 = GetSpan(p.n_layers * p.dim * p.hidden_dim);
    w.rms_final_weight = GetSpan(p.dim);
    position += p.seq_len * p.dim / p.n_heads; // skip what used to be freq_cis_real and freq_cis_imag (for RoPE)
    w.wcls = sharedWeights ? w.token_embedding_table : GetSpan(p.vocab_size * p.dim);
    return;

    float[] GetSpan(long len) {
        var buf = new float[len];
        a.ReadArray(position, buf, 0, (int)len);
        position += sizeof(float) * len;
        return buf;
    }
}

static Transformer BuildTransformer(Options options) {
    // read in the Config and the Weights from the checkpoint
    var configSize = Marshal.SizeOf(typeof(Config));
    using FileStream file = File.OpenRead(options.CheckpointPath);
    // read in the config header
    byte[] configBytes = new byte[configSize];
    if (file.Read(configBytes, 0, configBytes.Length) != configBytes.Length) { Environment.Exit(1); }
    // Convert byte array to Config struct
    var handle = GCHandle.Alloc(configBytes, GCHandleType.Pinned);
    var fileLength = file.Length;
    var t = new Transformer { config = (Config)Marshal.PtrToStructure(handle.AddrOfPinnedObject(), typeof(Config))! };
    t.config.vocab_size = Math.Abs(t.config.vocab_size);
    handle.Free();
    file.Close();

    using var fd = MemoryMappedFile.CreateFromFile(options.CheckpointPath, FileMode.Open);
    using var accessor = fd.CreateViewAccessor(configSize, fileLength - configSize);
    MemoryMapWeights(ref t.weights, ref t.config, accessor, t.config.vocab_size > 0);
    // allocate the RunState buffers
    t.state = BuildRunState(t.config);

    return t;
}

static void RmsNorm(Span<float> o, Span<float> x, Span<float> weight, int size) {
    // calculate sum of squares
    float ss = TensorPrimitives.SumOfSquares(x[..size]);
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / MathF.Sqrt(ss);
    // normalize and scale
    for (int j = 0; j < size; j++) o[j] = weight[j] * (ss * x[j]);
}

static void Softmax(Span<float> x) {
    // find max value (for numerical stability)
    float maxVal = TensorPrimitives.Max(x);
    // exp and sum
    float sum = 0.0f;
    for (int i = 0; i < x.Length; i++) {
        x[i] = MathF.Exp(x[i] - maxVal);
        sum += x[i];
    }
    // normalize
    TensorPrimitives.Divide(x, sum, x);
}

static void MatMul(IList<float> xout, ArraySegment<float> x, ArraySegment<float> w, int n, int d) =>
    // W (d,n) @ x (n,) -> xout (d,)
    Parallel.For(0, d, i => { xout[i] = TensorPrimitives.Dot(w.Slice(i * n, n), x); });

static Span<float> Forward(Transformer tr, int token, int pos) {
    // a few convenience variables
    int dim = tr.config.dim;
    int hidden_dim =  tr.config.hidden_dim;
    int head_size = dim / tr.config.n_heads;

    // copy the token embedding into x
    Array.Copy(tr.weights.token_embedding_table, token * dim, tr.state.x, 0, dim);

    // forward all the layers
    for (int l = 0; l < tr.config.n_layers; l++) {
        // attention rmsnorm
        RmsNorm(tr.state.xb, tr.state.x, tr.weights.rms_att_weight[(l * dim)..], dim);

        // qkv matmuls for this position
        MatMul(tr.state.q, tr.state.xb, tr.weights.wq[(l * dim * dim) ..], dim, dim);
        MatMul(tr.state.k, tr.state.xb, tr.weights.wk[(l * dim * dim) ..], dim, dim);
        MatMul(tr.state.v, tr.state.xb, tr.weights.wv[(l * dim * dim) ..], dim, dim);

        // RoPE relative positional encoding: complex-valued rotate q and k in each head
        for (int i = 0; i < dim; i += 2) {
            int head_dim = i % head_size;
            float freq = 1.0f / (float)Math.Pow(10000.0f, head_dim / (float)head_size);
            float val = pos * freq;
            float q0 = tr.state.q[i];
            float q1 = tr.state.q[i + 1];
            float k0 = tr.state.k[i];
            float k1 = tr.state.k[i + 1];
            float fcr = (float)Math.Cos(val);
            float fci = (float)Math.Sin(val);
            tr.state.q[i] = q0 * fcr - q1 * fci;
            tr.state.q[i + 1] = q0 * fci + q1 * fcr;
            tr.state.k[i] = k0 * fcr - k1 * fci;
            tr.state.k[i + 1] = k0 * fci + k1 * fcr;
        }
        
        // key and value point to the kv cache
        int loff = l * tr.config.seq_len * dim; // kv cache layer offset for convenience
        Array.Copy(tr.state.k, 0, tr.state.key_cache, loff + pos * dim, dim);
        Array.Copy(tr.state.v, 0, tr.state.value_cache, loff + pos * dim, dim);
        
        // multihead attention. iterate over all heads
        Parallel.For(0, tr.config.n_heads, h => {
            // get the query vector for this head
            int qOffset = h * head_size;

            // attention scores for this head
            Span<float> att = tr.state.att.AsSpan(h * tr.config.seq_len);

            // iterate over all timesteps, including the current one
            for (int t = 0; t <= pos; t++)
            {
                // get the key vector for this head and at this timestep
                int keyCacheOffset = loff + t * dim + h * head_size;

                // calculate the attention score as the dot product of q and k
                float score = 0.0f;
                for (int i = 0; i < head_size; i++) score += tr.state.q[i + qOffset] * tr.state.key_cache[i + keyCacheOffset];

                score /= MathF.Sqrt(head_size);

                // save the score to the attention buffer
                att[t] = score;
            }

            // softmax the scores to get attention weights, from 0..pos inclusively
            Softmax(att[..(pos + 1)]);

            // weighted sum of the values, store back into xb
            int xbOffset = h * head_size;
            for (int i = xbOffset; i < xbOffset + head_size; i++) tr.state.xb[i] = 0f;

            for (int t = 0; t <= pos; t++)
            {
                // get the value vector for this head and at this timestep
                int vOffset = loff + t * dim + h * head_size;

                // get the attention weight for this timestep
                float a = att[t];

                // accumulate the weighted value into xb
                for (int i = 0; i < head_size; i++)
                    tr.state.xb[i + xbOffset] += a * tr.state.value_cache[i + vOffset];
            }
        });

        // final matmul to get the output of the attention
        MatMul(tr.state.xb2, tr.state.xb, tr.weights.wo[(l * dim * dim)..], dim, dim);

        // residual connection back into x
        for (int i = 0; i < dim; i++) { tr.state.x[i] += tr.state.xb2[i]; }

        // ffn rmsnorm
        RmsNorm(tr.state.xb, tr.state.x, tr.weights.rms_ffn_weight[(l * dim)..], dim);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        MatMul(tr.state.hb, tr.state.xb, tr.weights.w1[(l * dim * hidden_dim)..], dim, hidden_dim);
        MatMul(tr.state.hb2, tr.state.xb, tr.weights.w3[(l * dim * hidden_dim)..], dim, hidden_dim);

        // SwiGLU non-linearity
        for (int i = 0; i < hidden_dim; i++) {
            float val = tr.state.hb[i];
            // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
            val *= (1.0f / (1.0f + MathF.Exp(-val)));
            // elementwise multiply with w3(x)
            val *= tr.state.hb2[i];
            tr.state.hb[i] = val;
        }

        // final matmul to get the output of the ffn
        MatMul(tr.state.xb, tr.state.hb, tr.weights.w2[(l * dim * hidden_dim)..], hidden_dim, dim);

        // residual connection
        for (int i = 0; i < dim; i++) { tr.state.x[i] += tr.state.xb[i]; }
    }

    // final rmsnorm
    RmsNorm(tr.state.x, tr.state.x, tr.weights.rms_final_weight, dim);

    // classifier into logits
    MatMul(tr.state.logits, tr.state.x, tr.weights.wcls, tr.config.dim, tr.config.vocab_size);
    return tr.state.logits;
}

static Tokenizer BuildTokenizer(int vocabSize, Options options) {
    var t = new Tokenizer {
        // i should have written the vocab_size into the tokenizer file... sigh
        vocab_size = vocabSize,
        // malloc space to hold the scores and the strings
        vocab = new string[vocabSize],
        vocab_scores = new float[vocabSize],
    };
    // read in the file
    using FileStream file = File.OpenRead(options.TokenizerPath);
    BinaryReader reader = new BinaryReader(file);
    try {
        t.max_token_length = (uint)reader.ReadInt32();
        for (int j = 0; j < vocabSize; j++) {
            t.vocab_scores[j] = reader.ReadSingle();
            var len = reader.ReadInt32();
            t.vocab[j] = Encoding.UTF8.GetString(reader.ReadBytes(len));
        }
    } catch (EndOfStreamException) {
        Console.Error.WriteLine("failed read");
        Environment.Exit(1);
    }

    return t;
}

static string Decode(ref Tokenizer t, int prev_token, int token) {
    string piece = t.vocab[token];
    // following BOS (1) token, sentencepiece decoder strips any leading whitespace (see PR #89)
    if (prev_token == 1 && piece[0] == ' ') { piece = piece[1..]; }
    // careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
    // parse this and convert and return the actual byte
    if (piece[0] == '<' && piece[1] == '0' && piece[2] == 'x' && piece[^1] == '>' 
        && byte.TryParse(piece[3..5], NumberStyles.HexNumber, null, out byte byte_val)) {
        piece = ((char)byte_val).ToString();
    }
    return piece;
}

static void SafePrint(string piece) {
    // piece might be a raw byte token, and we only want to print printable chars or whitespace
    // because some of the other bytes can be various control codes, backspace, etc.
    if (string.IsNullOrEmpty(piece)) { return; }
    if (piece.Length == 1) {
        char byteVal = piece[0];
        if (!(char.IsLetterOrDigit(byteVal) || char.IsWhiteSpace(byteVal))) {
            return; // bad byte, don't print it
        }
    }
    Console.Write(piece);
}

static int StrLookup(string str, string[] vocab, int vocabSize) {
    for (int i = 0; i < vocabSize; i++)
        if (str == vocab[i])
            return i;
    return -1;
}

static int Encode(ref Tokenizer t, string? text, byte bos, byte eos, ref int[] tokens) {
    // encode the string text (input) into an upper-bound preallocated tokens[] array
    // bos != 0 means prepend the BOS token (=1), eos != 0 means append the EOS token (=2)
    if (text == null) { Console.Error.WriteLine("cannot encode NULL text"); Environment.Exit(1); }
    
    // create a temporary buffer that will store merge candidates of always two consecutive tokens
    // *2 for concat, +1 for null terminator +2 for UTF8 (in case max_token_length is 1)
    var strBuffer = new StringBuilder((int)t.max_token_length * 2 + 1);

    // start at 0 tokens
    int n_tokens = 0;

    // add optional BOS (=1) token, if desired
    if (bos != 0) tokens[n_tokens++] = 1;

    foreach (char c in text)
    {
        strBuffer.Clear();
        strBuffer.Append(c);

        int id = StrLookup(strBuffer.ToString(), t.vocab, t.vocab_size);
        if (id == -1) { throw new Exception("Encoding error"); }

        tokens[n_tokens] = id;
        n_tokens++;
    }

    // merge the best consecutive pair each iteration, according to the scores in vocab_scores
    while (true) {
        float bestScore = float.MinValue;
        int bestId = -1;
        int bestIdx = -1;

        for (int i = 0; i < n_tokens - 1; i++) {
            // check if we can merge the pair (tokens[i], tokens[i+1])
            strBuffer.Clear();
            strBuffer.Append(t.vocab[tokens[i]]).Append(t.vocab[tokens[i + 1]]);
            int id = StrLookup(strBuffer.ToString(), t.vocab, t.vocab_size);
            if (id != -1 && t.vocab_scores[id] > bestScore) {
                // this merge pair exists in vocab! record its score and position
                bestScore = t.vocab_scores[id];
                bestId = id;
                bestIdx = i;
            }
        }

        if (bestIdx == -1) break; // we couldn't find any more pairs to merge, so we're done

        // merge the consecutive pair (bestIdx, bestIdx+1) into new token bestId
        tokens[bestIdx] = bestId;
        // delete token at position bestIdx+1, shift the entire sequence back 1
        for (int i = bestIdx + 1; i < n_tokens - 1; i++) tokens[i] = tokens[i + 1];
        n_tokens--; // token length decreased
    }

    // add optional EOS (=2) token, if desired
    if (eos != 0) tokens[n_tokens++] = 2;

    return n_tokens;
}

static int SampleArgmax(float[] probabilities, int n) {
    // return the index that has the highest probability
    int maxI = 0;
    float maxP = probabilities[0];
    for (int i = 1; i < n; i++) {
        if (probabilities[i] > maxP) {
            maxI = i;
            maxP = probabilities[i];
        }
    }
    return maxI;
}

static int SampleMult(float[] probabilities, int n, float coin) {
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from random_f32()
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1; // in case of rounding errors
}

static int Compare(ProbIndex a, ProbIndex b) {
    if (a.prob > b.prob) return -1;
    if (a.prob < b.prob) return 1;
    return 0;
}

static int SampleTopp(float[] probabilities, int n, float topp, ProbIndex[] probindex, float coin) {
    // top-p sampling (or "nucleus sampling") samples from the smallest set of
    // tokens that exceed probability topp. This way we never sample tokens that
    // have very low probabilities and are less likely to go "off the rails".
    // coin is a random number in [0, 1), usually from random_f32()

    int n0 = 0;
    // quicksort indices in descending order of probabilities
    // values smaller than (1 - topp) / (n - 1) cannot be part of the result
    // so for efficiency we crop these out as candidates before sorting
    float cutoff = (1.0f - topp) / (n - 1);
    for (int i = 0; i < n; i++) {
        if (probabilities[i] >= cutoff) {
            probindex[n0].index = i;
            probindex[n0].prob = probabilities[i];
            n0++;
        }
    }
    Array.Sort(probindex, 0, n0, Comparer<ProbIndex>.Create(Compare));

    // truncate the list where cumulative probability exceeds topp
    float cumulative_prob = 0.0f;
    int last_idx = n0 - 1; // in case of rounding errors consider all elements
    for (int i = 0; i < n0; i++) {
        cumulative_prob += probindex[i].prob;
        if (cumulative_prob > topp) {
            last_idx = i;
            break; // we've exceeded topp by including last_idx
        }
    }

    // sample from the truncated list
    float r = coin * cumulative_prob;
    float cdf = 0.0f;
    for (int i = 0; i <= last_idx; i++) {
        cdf += probindex[i].prob;
        if (r < cdf) {
            return probindex[i].index;
        }
    }
    return probindex[last_idx].index; // in case of rounding errors
}

static Sampler BuildSampler(int vocabSize, Options opts) => new() {
    vocab_size = vocabSize,
    temperature = opts.Temperature,
    topp = opts.Topp,
    rng_state = opts.Seed,
    // buffer only used with nucleus sampling; may not need but it's ~small
    probindex = new ProbIndex[vocabSize]
};

static uint RandomU32(ref ulong state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    return (uint)((state * 0x2545F4914F6CDD1Dul) >> 32);
}

static float RandomF32(ref ulong state) { // random float32 in [0,1)
    return (RandomU32(ref state) >> 8) / 16777216.0f;
}

static int Sample(ref Sampler sampler, float[] logits) {
    // sample the token given the logits and some hyperparameters
    int next;
    if (sampler.temperature == 0.0f) {
        // greedy argmax sampling: take the token with the highest probability
        next = SampleArgmax(logits, sampler.vocab_size);
    } else {
        // apply the temperature to the logits
        for (int q = 0; q < sampler.vocab_size; q++) { logits[q] /= sampler.temperature; }
        // apply softmax to the logits to get the probabilities for next token
        Softmax(new Span<float>(logits, 0, sampler.vocab_size));
        // flip a (float) coin (this is our source of entropy for sampling)
        float coin = RandomF32(ref sampler.rng_state);
        // we sample from this distribution to get the next token
        if (sampler.topp <= 0 || sampler.topp >= 1) {
            // simply sample from the predicted probability distribution
            next = SampleMult(logits, sampler.vocab_size, coin);
        } else {
            // top-p (nucleus) sampling, clamping the least likely tokens to zero
            next = SampleTopp(logits, sampler.vocab_size, sampler.topp, sampler.probindex, coin);
        }
    }
    return next;
}

static void Generate(ref Transformer transformer, ref Tokenizer tokenizer, ref Sampler sampler, Options options) {
    // encode the (string) prompt into tokens sequence
    int[] promptTokens = new int[options.Prompt.Length + 3]; // +3 for '\0', ?BOS, ?EOS
    int numPromptTokens = Encode(ref tokenizer, options.Prompt, 1, 0, ref promptTokens);
    if (numPromptTokens < 1) {
        Console.Error.WriteLine("something is wrong, expected at least 1 prompt token");
        Environment.Exit(1);
    }

    // start the main loop
    var sw = new Stopwatch();    // used to time our code, only initialized after first iteration
    int token = promptTokens[0]; // kick off with the first token in the prompt
    int pos = 0;     // position in the sequence
    while (pos < options.Steps) {

        // forward the transformer to get logits for the next token
        Span<float> logits = Forward(transformer, token, pos);

        // advance the state machine
        var next = // will store the next token in the sequence
            // if we are still processing the input prompt, force the next prompt token
            // otherwise sample the next token from the logits
            pos < numPromptTokens - 1 ? promptTokens[pos + 1] : Sample(ref sampler, logits.ToArray());
        pos++;

        // data-dependent terminating condition: the BOS (=1) token delimits sequences
        if (next == 1) { break; }

        // print the token as string, decode it with the Tokenizer object
        string piece = Decode(ref tokenizer, token, next);
        SafePrint(piece); // same as Console.Write(piece), but skips "unsafe" bytes
        Console.Out.Flush();
        token = next;

        // init the timer here because the first iteration can be slower
        if (!sw.IsRunning) { sw.Start(); }
    }
    Console.WriteLine();

    // report achieved tok/s (pos-1 because the timer starts after first iteration)
    if (pos > 1) {
        sw.Stop();
        Log($"achieved tok/s: {(pos-1) / sw.Elapsed.TotalSeconds}");
    }
}

static void Chat(ref Transformer transformer, ref Tokenizer tokenizer, ref Sampler sampler, Options options)
{
    // buffers for reading the system prompt and user prompt from stdin
    string systemPrompt = "";
    int numPromptTokens = 0;
    int[] promptTokens = new int[1152];
    int userIdx = 0;

    // start the main loop
    bool userTurn = true; // user starts
    int next = 0;        // will store the next token in the sequence
    int pos = 0;     // position in the sequence
    while (pos < options.Steps) {
        // when it is the user's turn to contribute tokens to the dialog...
        if (userTurn) {
            // get the (optional) system prompt at position 0
            if (pos == 0) {
                // at position 0, the user can also contribute a system prompt
                if (options.SystemPrompt == null) {
                    // system prompt was not passed in, attempt to get it from stdin
                    Console.Write("Enter system prompt (optional): ");
                    systemPrompt = Console.ReadLine()!;
                }
                else {
                    // system prompt was passed in, use it
                    systemPrompt = options.SystemPrompt;
                }
            }
            // get the user prompt
            string userPrompt;
            if (pos == 0 && options.Prompt != null) {
                // user prompt for position 0 was passed in, use it
                userPrompt = options.Prompt;
            }
            else {
                // otherwise get user prompt from stdin
                Console.Write("User: ");
                userPrompt = Console.ReadLine()!;
            }
            // render user/system prompts into the Llama 2 Chat schema
            string renderedPrompt;
            if (pos == 0 && systemPrompt[0] != '\0') {
                const string systemTemplate = "[INST] <<SYS>>\n{0}\n<</SYS>>\n\n{1} [/INST]";
                renderedPrompt = string.Format(systemTemplate, systemPrompt, userPrompt);
            }
            else {
                string userTemplate = "[INST] {0} [/INST]";
                renderedPrompt = string.Format(userTemplate, userPrompt);
            }
            // encode the rendered prompt into tokens
            numPromptTokens = Encode(ref tokenizer, new string(renderedPrompt), 1, 0, ref promptTokens);
            userIdx = 0; // reset the user index
            userTurn = false;
            Console.Write("Assistant: ");
        }

        // determine the token to pass into the transformer next
        // if we are still processing the input prompt, force the next prompt token
            // otherwise use the next token sampled from previous turn
        var token = userIdx < numPromptTokens ? promptTokens[userIdx++] : next;       // stores the current token to feed into the transformer
        // EOS (=2) token ends the Assistant turn
        if (token == 2) { userTurn = true; }

        // forward the transformer to get logits for the next token
        Span<float> logits = Forward(transformer, token, pos);
        next = Sample(ref sampler, logits.ToArray());
        pos++;

        if (userIdx >= numPromptTokens && next != 2) {
            // the Assistant is responding, so print its output
            string piece = Decode(ref tokenizer, token, next);
            SafePrint(piece); // same as Console.Write(piece), but skips "unsafe" bytes
            Console.Out.Flush();
        }
        if (next == 2) { Console.WriteLine(); }
    }
    Console.WriteLine();
}

static void Log(string entry) => Console.WriteLine($"[{DateTime.UtcNow:HH:mm:ss}] {entry}");

[StructLayout(LayoutKind.Sequential)]
struct ProbIndex {
    public float prob;
    public int index;
}

[StructLayout(LayoutKind.Sequential)]
struct Sampler {
    public int vocab_size;
    public ProbIndex[] probindex; // buffer used in top-p sampling
    public float temperature;
    public float topp;
    public ulong rng_state;
}

[StructLayout(LayoutKind.Sequential)]
struct Tokenizer {
    public string[] vocab;
    public float[] vocab_scores;
    public int vocab_size;
    public uint max_token_length;
}

[StructLayout(LayoutKind.Sequential)]
struct Config {
    public int dim; // transformer dimension
    public int hidden_dim; // for ffn layers
    public int n_layers; // number of layers
    public int n_heads; // number of query heads
    public int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    public int vocab_size; // vocabulary size, usually 256 (byte-level)
    public int seq_len; // max sequence length
}

[StructLayout(LayoutKind.Sequential)]
struct TransformerWeights {
    // token embedding table
    public float[] token_embedding_table; // (vocab_size, dim)
    // weights for rmsnorms
    public ArraySegment<float> rms_att_weight; // (layer, dim) rmsnorm weights
    public float[] rms_ffn_weight; // (layer, dim)
    // weights for matmuls. note dim == n_heads * head_size
    public ArraySegment<float> wq; // (layer, dim, n_heads * head_size)
    public ArraySegment<float> wk; // (layer, dim, n_kv_heads * head_size)
    public ArraySegment<float> wv; // (layer, dim, n_kv_heads * head_size)
    public ArraySegment<float> wo; // (layer, n_heads * head_size, dim)
    // weights for ffn
    public ArraySegment<float> w1; // (layer, hidden_dim, dim)
    public ArraySegment<float> w2; // (layer, dim, hidden_dim)
    public ArraySegment<float> w3; // (layer, hidden_dim, dim)
    // final rmsnorm
    public float[] rms_final_weight; // (dim,)
    // (optional) classifier weights for the logits, on the last layer
    public float[] wcls; 
}

[StructLayout(LayoutKind.Sequential)]
struct RunState {
    // current wave of activations
    public float[] x; // activation at current time stamp (dim,)
    public float[] xb; // same, but inside a residual branch (dim,)
    public float[] xb2; // an additional buffer just for convenience (dim,)
    public float[] hb; // buffer for hidden dimension in the ffn (hidden_dim,)
    public float[] hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
    public float[] q; // query (dim,)
    public float[] k; // key (dim,)
    public float[] v; // value (dim,)
    public float[] att; // buffer for scores/attention values (n_heads, seq_len)
    public float[] logits; // output logits
    // kv cache
    public float[] key_cache;   // (layer, seq_len, dim)
    public float[] value_cache; // (layer, seq_len, dim)
}

[StructLayout(LayoutKind.Sequential)]
struct Transformer {
    public Config config; // the hyperparameters of the architecture (the blueprint)
    public TransformerWeights weights; // the weights of the model
    public RunState state; // buffers for the "wave" of activations in the forward pass
}

enum Mode {Generate,Chat}
// ReSharper disable once ClassNeverInstantiated.Global
class Options {
    [Value(0, MetaName = "checkpoint", Required = true, HelpText = "Path to the model checkpoint file")]
    // ReSharper disable once UnusedAutoPropertyAccessor.Global
    public required string CheckpointPath { get; set; }
    [Option('t', "temperature", Required = false, HelpText = "Temperature in [0,inf], default 1.0")]
    public float Temperature { get; set; } = 1.0f;
    [Option('p', "topp", Required = false, HelpText = "P value in top-p (nucleus) sampling in [0,1], default 0.9")]
    public float Topp { get; set; } = 0.9f;
    [Option('s', "seed", Required = false, HelpText = "Random seed, default time(NULL)")]
    public ulong Seed { get; set; }
    [Option('n', "steps", Required = false, HelpText = "Number of steps to run for, default 256. 0 = max_seq_len")]
    public int Steps { get; set; } = 256;
    [Option('i', "input", Required = false, HelpText = "Input prompt")]
    public string Prompt { get; set; } = "Once upon a time";
    [Option('z', "tokenizer", Required = false, HelpText = "Optional path to custom tokenizer")]
    public string TokenizerPath { get; set; } = "tokenizer.bin";
    [Option('m', "mode", Required = false, HelpText = "Mode: generate|chat, default: generate")]
    public Mode Mode { get; set; } = Mode.Generate;
    [Option('y', "systemPrompt", Required = false, HelpText = "(Optional) system prompt in chat mode")]
    public string? SystemPrompt { get; set; } = null;
    [Option('v', "verbose", Required = false, HelpText = "(Optional) system prompt in chat mode")]
    public bool Verbose { get; set; } = false;
}
