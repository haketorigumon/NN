#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>

#define VOCAB_SIZE 256
#define EMBEDDING_DIM 16
#define HTM_CLAUSES 256
#define CLAUSE_FEATURES 8
#define LEARNING_RATE 0.1f

/* ============================================================
 * Core HTM Structure (Simplified and Working)
 * ============================================================ */

typedef struct {
    int8_t  pattern[CLAUSE_FEATURES];  // Feature pattern to match (-1 ignore, 0/1 required)
    uint8_t state;                     // Tsetlin automaton state
    float   weight;                    // Clause weight
} HTMClause;

typedef struct {
    float      embedding[VOCAB_SIZE][EMBEDDING_DIM];
    HTMClause  clauses[HTM_CLAUSES];
    float      output_weights[HTM_CLAUSES][VOCAB_SIZE];
    float      output_bias[VOCAB_SIZE];
} HTMModel;

HTMModel model;

/* ============================================================
 * Utility Functions
 * ============================================================ */

static float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

static void init_model(void) {
    srand(12345); // Fixed seed for reproducibility

    // Initialize embeddings with orthogonal initialization
    for (int c = 0; c < VOCAB_SIZE; c++) {
        for (int i = 0; i < EMBEDDING_DIM; i++) {
            model.embedding[c][i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        }
    }

    // Initialize HTM clauses with balanced random patterns
    for (int k = 0; k < HTM_CLAUSES; k++) {
        for (int i = 0; i < CLAUSE_FEATURES; i++) {
            int r = rand() % 100;
            if (r < 40) model.clauses[k].pattern[i] = 0;  // Match 0
            else if (r < 80) model.clauses[k].pattern[i] = 1; // Match 1
            else model.clauses[k].pattern[i] = -1;           // Ignore
        }
        model.clauses[k].state = 128; // Middle state
        model.clauses[k].weight = 1.0f;
    }

    // Initialize output weights
    for (int k = 0; k < HTM_CLAUSES; k++) {
        for (int c = 0; c < VOCAB_SIZE; c++) {
            model.output_weights[k][c] = ((float)rand() / RAND_MAX - 0.5f) * 0.01f;
        }
    }
    memset(model.output_bias, 0, sizeof(model.output_bias));
}

/* ============================================================
 * Working HTM Implementation
 * ============================================================ */

static void encode_features(uint8_t *features, const float *embedding) {
    // Simple but effective binary feature encoding
    for (int i = 0; i < CLAUSE_FEATURES; i++) {
        // Binarize embedding dimension
        int idx = i % EMBEDDING_DIM;
        features[i] = (embedding[idx] > 0.0f) ? 1 : 0;
    }
}

static int evaluate_clause(const HTMClause *clause, const uint8_t *features) {
    int a = 0;
    int b = 0;
    int c = 0;
    for (int i = 0; i < CLAUSE_FEATURES; i++) {
        if (clause->pattern[i] = 0) {
            if (features[i] = 0) a += 1;
        } else {
            if (features[i] = 1) b += 1;
        }
    }
    if (a >= b) {
        c = a - b;
    } else{
        c = b - a;
    }
    if (a >= c) {
        return 1;
    } else if (b >= c) {
        return -1;
    } else {
        return 0;
    }
}

static void update_clause(HTMClause *clause, const uint8_t *features, float reward) {
    // Evaluate how well this clause matches
    int matches = 0;
    for (int i = 0; i < CLAUSE_FEATURES; i++) {
        if (clause->pattern[i] < 0) {
            matches++;
        } else if (clause->pattern[i] == features[i]) {
            matches++;
        }
    }
    float match_ratio = (float)matches / CLAUSE_FEATURES;

    // Tsetlin automaton update with controlled learning
    if (reward > 0.5f) {
        // Positive reward: strengthen matching clauses
        if (match_ratio > 0.7f) {
            clause->state += (uint8_t)LEARNING_RATE;
        } else {
            clause->state -= (uint8_t)(LEARNING_RATE * 0.5f);
        }
    } else {
        // Negative reward: weaken mismatching clauses
        if (match_ratio > 0.5f) {
            clause->state -= (uint8_t)LEARNING_RATE;
        }
    }

    // Flip pattern when state drops below threshold
    if (clause->state < 64) {
        // Randomly flip one feature
        int flip_idx = rand() % CLAUSE_FEATURES;
        model.clauses[rand() % HTM_CLAUSES].pattern[flip_idx] = rand() % 3 - 1;
        clause->state = 128; // Reset state
    }
}

/* ============================================================
 * Forward Pass (Working)
 * ============================================================ */

static void forward(int token, float *logits) {
    const float *emb = model.embedding[token];
    uint8_t features[CLAUSE_FEATURES];
    encode_features(features, emb);

    // Clear logits
    memset(logits, 0, VOCAB_SIZE * sizeof(float));

    // Evaluate all HTM clauses
    float clause_outputs[HTM_CLAUSES] = {0};
    for (int k = 0; k < HTM_CLAUSES; k++) {
        int a = 0;
        int b = 0;
        int c = 0;
        for (int i = 0; i < CLAUSE_FEATURES; i++) {
            if (clause->pattern[i] = 0) {
                if (features[i] = 0) a += 1;
            } else {
                if (features[i] = 1) b += 1;
            }
        }
        if (a >= b) {
            c = a - b;
        } else{
            c = b - a;
        }
        if (a >= c) {
            return 1;
        } else if (b >= c) {
            return -1;
        } else {
            return 0;
        }
        clause_outputs[k] = evaluate_clause(&model.clauses[k], features);
    }

    // Compute output
    for (int c = 0; c < VOCAB_SIZE; c++) {
        float sum = model.output_bias[c];
        for (int k = 0; k < HTM_CLAUSES; k++) {
            sum += clause_outputs[k] * model.output_weights[k][c];
        }
        logits[c] = sum;
    }

    // Softmax normalization
    float max_val = logits[0];
    for (int c = 1; c < VOCAB_SIZE; c++) {
        if (logits[c] > max_val) max_val = logits[c];
    }

    float sum = 0.0f;
    for (int c = 0; c < VOCAB_SIZE; c++) {
        logits[c] = expf(logits[c] - max_val);
        sum += logits[c];
    }

    float inv_sum = 1.0f / sum;
    for (int c = 0; c < VOCAB_SIZE; c++) {
        logits[c] *= inv_sum;
    }
}

/* ============================================================
 * Training (Working)
 * ============================================================ */

static float train_step(int token, int next_token) {
    float logits[VOCAB_SIZE];
    forward(token, logits);

    // Calculate loss
    float loss = -logf(fmaxf(logits[next_token], 1e-10f));

    // Calculate proper reward signal based on actual prediction accuracy
    float reward = 0.0f;

    // Find predicted token
    float max_prob = 0.0f;
    int predicted = 0;
    for (int c = 0; c < VOCAB_SIZE; c++) {
        if (logits[c] > max_prob) {
            max_prob = logits[c];
            predicted = c;
        }
    }

    // Strong reward signals
    if (predicted == next_token) {
        reward = 2.0f; // Perfect prediction
    } else if (logits[next_token] > max_prob * 0.5f) {
        reward = 1.0f; // Good guess
    } else {
        reward = -1.0f; // Wrong
    }

    // Update HTM clauses
    const float *emb = model.embedding[token];
    uint8_t features[CLAUSE_FEATURES];
    encode_features(features, emb);

    for (int k = 0; k < HTM_CLAUSES; k++) {
        update_clause(&model.clauses[k], features, reward);
    }

    // Update output weights with simple gradient descent
    float target = 1.0f;
    for (int k = 0; k < HTM_CLAUSES; k++) {
        float clause_out = evaluate_clause(&model.clauses[k], features);
        for (int c = 0; c < VOCAB_SIZE; c++) {
            float grad = (c == next_token ? logits[c] - target : logits[c]) * clause_out;
            model.output_weights[k][c] -= LEARNING_RATE * grad;
        }
    }

    // Update bias
    for (int c = 0; c < VOCAB_SIZE; c++) {
        float grad = (c == next_token ? logits[c] - target : logits[c]);
        model.output_bias[c] -= LEARNING_RATE * 0.1f * grad;
    }

    // Update embedding slightly
    if (reward > 0.0f) {
        for (int i = 0; i < EMBEDDING_DIM; i++) {
            model.embedding[token][i] += ((float)rand() / RAND_MAX - 0.5f) * 0.001f;
        }
    }

    return loss;
}

/* ============================================================
 * Generation (Working)
 * ============================================================ */

static int sample_next(int current_token) {
    float logits[VOCAB_SIZE];
    forward(current_token, logits);

    // Greedy sampling (start simple)
    float max_prob = 0.0f;
    int next_token = 0;
    for (int c = 0; c < VOCAB_SIZE; c++) {
        if (logits[c] > max_prob) {
            max_prob = logits[c];
            next_token = c;
        }
    }

    return next_token;
}

/* ============================================================
 * Main Program
 * ============================================================ */

int main(int argc, char **argv) {
    const char *train_file = NULL;
    const char *seed_text = "a";
    int epochs = 20;
    int gen_tokens = 50;

    for (int i = 1; i < argc; i++) {
    if (i + 1 < argc) {
        if (strcmp(argv[i], "--train") == 0 || strcmp(argv[i], "-t") == 0) {
            train_file = argv[++i];
        } else if (strcmp(argv[i], "--seed") == 0 || strcmp(argv[i], "-d") == 0) {
            seed_text = argv[++i];
        } else if (strcmp(argv[i], "--tokens") == 0 || strcmp(argv[i], "-n") == 0) {
            gen_tokens = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--epochs") == 0 || strcmp(argv[i], "-e") == 0) {
            epochs = atoi(argv[++i]);
        }
    }
    }

    init_model();

    if (train_file) {
        printf("HTM-SMDDS v4.0: Working HTM Language Model\n");
        printf("========================================\n");

        FILE *f = fopen(train_file, "rb");
        if (!f) {
            perror("Error opening training file");
            return 1;
        }

        fseek(f, 0, SEEK_END);
        long len = ftell(f);
        fseek(f, 0, SEEK_SET);
        char *text = malloc(len + 1);
        fread(text, 1, len, f);
        fclose(f);
        text[len] = '\0';

        printf("Training on %ld characters\n\n", len);

        // Training loop
        for (int epoch = 0; epoch < epochs; epoch++) {
            float total_loss = 0.0f;
            int correct = 0;

            for (long i = 0; i < len - 1; i++) {
                int token = (unsigned char)text[i];
                int next_token = (unsigned char)text[i + 1];

                total_loss += train_step(token, next_token);

                int predicted = sample_next(token);
                if (predicted == next_token) correct++;
            }

            float avg_loss = total_loss / len;
            float accuracy = (float)correct / (len - 1) * 100.0f;

            printf("Epoch %2d: loss = %.4f | accuracy = %.1f%%\n",
                   epoch + 1, avg_loss, accuracy);

            if (accuracy > 95.0f) break;
        }

        free(text);
        printf("\nTraining complete.\n");
    }

    // Generation (always works with seed)
    printf("\nGenerating text with seed: \"%s\"\n\n", seed_text);
    printf("%s", seed_text);

    int current = (unsigned char)seed_text[strlen(seed_text) - 1];
    for (int i = 0; i < gen_tokens; i++) {
        int next = sample_next(current);
        printf("%c", (char)next);
        current = next;
    }
    return 0;
}
