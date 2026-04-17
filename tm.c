#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>

#define VOCAB_SIZE 256
#define EMBEDDING_DIM 16
#define CLAUSES 256
#define OUT_CLAUSES 256
#define META_CLAUSES 256
#define MEM 256
#define CLAUSE_FEATURES 8
#define META_CLAUSE_FEATURES CLAUSES * 2
#define TMS CLAUSE_FEATURES

#define BIT_ARRAY_SIZE 100
uint8_t bit_array[(BIT_ARRAY_SIZE + 7) / 8];

#define SET_BIT(arr, n)     ((arr)[(n)/8] |=  (1U << ((n)%8)))
#define CLEAR_BIT(arr, n)   ((arr)[(n)/8] &= ~(1U << ((n)%8)))
#define GET_BIT(arr, n)     (((arr)[(n)/8] & (1U << ((n)%8))) != 0)
#define TOGGLE_BIT(arr, n)  ((arr)[(n)/8] ^= (1U << ((n)%8)))

typedef struct {
    uint8_t  pattern[CLAUSE_FEATURES];
    uint8_t pattern_2[MEM];
    uint8_t pattern_3[MEM];
} Clause;

Clause clause;

static void init_model(void) {
    srand(12345);

    for (int k = 0; k < CLAUSES; k++) {
        for (int i = 0; i < CLAUSE_FEATURES; i++) {
            if (rand() / RAND_MAX >= 1 / 2) pattern[i] = 1;
        }
    }
}
static int categorical_sample(const double* probs, int num_classes)
{
    double cum = 0.0;
    double max_val = probs[0];
    for (int i = 1; i < num_classes; i++) {
        if (probs[i] > max_val) {
            max_val = probs[i];
        }
    }
    double s = 0.0;
    for (int i = 0; i < num_classes; i++) {
        s += max_val - probs[i];
    }
    s = s / num_classes;
    double a[num_classes];
    int k = 0.0;
    for (int i = 0; i < num_classes; i++) {
        if (probs[i] > s) {
            a[k] = i;
            k++;
        }
    }
    double p = 0.0;
    for (int i = 0; i < k; i++) {
        p += probs[a[i]];
    }
    double r = (double)rand() / RAND_MAX * p;
    for (int i = 0; i < k; i++) {
        cum += probs[a[i]];
        if (r <= cum) {
            return i;
        }
    }
    return num_classes - 1;
}

static void clauses(uint8_t *clause_outputs, const uint8_t *features, uint8_t *pattern, size_t clause_size, size_t feature_size) {
    for (int k = 0; k < clause_size; k++) {
            int a = 0;
            int b = 0;
            for (int i = 0; i < feature_size; i++) {
                if (GET_BIT(pattern,k*feature_size+i)) {
                    if (GET_BIT(features,i)) {
                        a += 1;
                    } else {
                        b += 1;
                    }
                }
                if (GET_BIT(pattern,k*feature_size+i+clause_size*feature_size)) {
                    if (GET_BIT(features,i)) {
                        b += 1;
                    } else {
                        a += 1;
                    }
                }
            }
            if (b < abs(a - b)) {
                SET_BIT(clause_outputs,k);
            }
    }
}
static int forward(uint8_t token, uint8_t *mem) {
    uint8_t *features = &token;
    uint8_t clause_outputs[(CLAUSES + 7) / 8] = {0};
    clauses(clause_outputs, features, pattern, CLAUSES, CLAUSE_FEATURES);
    uint8_t meta_clause_outputs[(META_CLAUSES * CLAUSE_FEATURES * 2 + 7) / 8] = {0};
    clauses(meta_clause_outputs, mem, pattern_2, META_CLAUSES * CLAUSE_FEATURES * 2, MEM);
    uint8_t hyper_clause_outputs[(MEM + 7) / 8] = {0};
    clauses(hyper_clause_outputs, clause_outputs, meta_clause_outputs, MEM, CLAUSES);
    int logits[VOCAB_SIZE];
    
    for (int k = 0; k < VOCAB_SIZE; k++) {
        int a = 0;
        uint8_t *r_clause_outputs = clauses(hyper_clause_outputs, pattern_3+k*MEM, tok, MEM);
        for (int i = 0; i < EMBEDDING_DIM; i++) {
            if(GET_BIT(r_clause_outputs,i)) {
                a++;
            }
        }
        logits[k] = a;
    }
    srand(time(NULL));
    double probs[VOCAB_SIZE];
    
    double max_val = logits[0];
    for (int i = 1; i < n; i++) {
        if (logits[i] > max_val) {
            max_val = logits[i];
        }
    }
    
    double sum = 0.0;
    for (int i = 0; i < VOCAB_SIZE; i++) {
        probs[i] = exp((double)(logits[i] - max_val));
        sum += probs[i];
    }
    for (int i = 0; i < VOCAB_SIZE; i++) {
        probs[i] /= sum;
    }
    
    int idx = categorical_sample(probs, VOCAB_SIZE);
    return idx;
}

static float train(uint8_t token, uint8_t next_token) {
    uint8_t *features = &token;
    uint8_t *clause_outputs = clauses(features, pattern, CLAUSES, CLAUSE_FEATURES);
    uint8_t *meta_clause_outputs = clauses(mem, pattern_2, META_CLAUSES * CLAUSE_FEATURES * 2, MEM);
    uint8_t *hyper_clause_outputs = clauses(clause_outputs, meta_clause_outputs, MEM, CLAUSES);
    int logits[VOCAB_SIZE];
    uint8_t *r_clause_outputs = malloc((size_t)VOCAB_SIZE * OUT_CLAUSES);
    if (r_clause_outputs == NULL) {
        perror("malloc for r_clause_outputs failed");
    }
    for (int k = 0; k < VOCAB_SIZE; k++) {
        int a = 0;
        clauses(r_clause_outputs + (size_t)k * OUT_CLAUSES, hyper_clause_outputs, pattern_3+k*MEM, OUT_CLAUSES, MEM);
        for (int i = 0; i < EMBEDDING_DIM; i++) {
            if(GET_BIT(r_clause_outputs,i)) {
                a++;
            }
        }
        logits[k] = a;
    }
    srand(time(NULL));
    double probs[VOCAB_SIZE];
    
    double max_val = logits[0];
    for (int i = 1; i < n; i++) {
        if (logits[i] > max_val) {
            max_val = logits[i];
        }
    }
    
    double sum = 0.0;
    for (int i = 0; i < VOCAB_SIZE; i++) {
        probs[i] = exp((double)(logits[i] - max_val));
        sum += probs[i];
    }
    for (int i = 0; i < VOCAB_SIZE; i++) {
        probs[i] /= sum;
    }
    float loss = 1 - probs[next_token];
    
    int idx = categorical_sample(probs, VOCAB_SIZE);
    if (next_token != idx) {
        a;
    }
    float loss = -logf(fmaxf(logits[next_token], 1e-10f));
    return loss;
}

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
        
        FILE *f = fopen(train_file, "rb");
        if (!f) {
            perror("Error opening training file");
            return 1;
        }

        fseek(f, 0, SEEK_END);
        long len = ftell(f);
        fseek(f, 0, SEEK_SET);
        uint8_t *text = malloc(len + 1);
        fread(text, 1, len, f);
        fclose(f);
        text[len] = '\0';

        printf("Training on %ld characters\n\n", len);

        for (int epoch = 0; epoch < epochs; epoch++) {
            float total_loss = 0.0f;
            int correct = 0;

            for (long i = 0; i < len - 1; i++) {
                uint8_t token = text[i];
                uint8_t next_token = text[i + 1];

                total_loss += train(token, next_token);

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
