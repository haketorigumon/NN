#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>

#define MEM_FILE "tmlm.mem"
#define WEIGHTS_FILE "tmlm.weights"

#define VOCAB_SIZE 256
#define INPUT_DIM 8
#define MEM 512

#define BLOCKS_OF_MEM_LAYER MEM
#define CLAUSES_PER_MEM_BLOCK 512
#define CLAUSES_OF_MEM_LAYER BLOCKS_OF_MEM_LAYER * CLAUSES_PER_MEM_BLOCK
#define FEATURES_PER_CLAUSE_OF_MEM_LAYER MEM
#define FEATURES_OF_MEM_LAYER FEATURES_PER_CLAUSE_OF_MEM_LAYER * CLAUSES_OF_MEM_LAYER

#define BLOCKS_OF_META_LAYER MEM
#define CLAUSES_PER_META_BLOCK INPUT_DIM * 2
#define FEATURES_PER_CLAUSE_OF_META CLAUSES_PER_MEM_BLOCK
#define FEATURES_PER_META_BLOCK CLAUSES_PER_META_BLOCK * FEATURES_PER_CLAUSE_OF_META
#define CLAUSES_OF_META_LAYER BLOCKS_OF_META_LAYER * CLAUSES_PER_META_BLOCK
#define FEATURES_OF_META_LAYER FEATURES_PER_CLAUSE_OF_META * CLAUSES_OF_META_LAYER


#define CLAUSES_OF_INPUT_LAYER MEM
#define FEATURES_PER_CLAUSE_OF_INPUT_LAYER INPUT_DIM
#define FEATURES_OF_INPUT_LAYER FEATURES_PER_CLAUSE_OF_INPUT_LAYER * CLAUSES_OF_INPUT_LAYER

#define CLAUSES_OF_OUT_LAYER 256
#define FEATURES_PER_CLAUSE_OF_OUT MEM
#define FEATURES_OF_OUT_LAYER FEATURES_PER_CLAUSE_OF_OUT * CLAUSES_OF_OUT_LAYER

#define CLASSES VOCAB_SIZE
#define CLAUSES_PER_CLASS 256
#define CLAUSES_OF_CLASS_LAYER CLASSES * CLAUSES_PER_CLASS
#define FEATURES_PER_CLAUSE_OF_CLASS MEM


#define SET_BIT(arr, n)     ((arr)[(n)/8] |=  (1U << ((n)%8)))
#define CLEAR_BIT(arr, n)   ((arr)[(n)/8] &= ~(1U << ((n)%8)))
#define GET_BIT(arr, n)     (((arr)[(n)/8] & (1U << ((n)%8))) != 0)
#define TOGGLE_BIT(arr, n)  ((arr)[(n)/8] ^= (1U << ((n)%8)))

uint8_t pattern[(FEATURES_OF_MEM_LAYER * 2 + 7) / 8];
uint8_t pattern_2[(FEATURES_PER_META_BLOCK * 2 + 7) / 8];
uint8_t pattern_3[(FEATURES_OF_INPUT_LAYER + 7) / 8];                               

uint8_t mem[(MEM + 7) / 8];

static void save_weights(const char *filename) {
    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        perror("保存权重文件失败");
        return;
    }
    fwrite(pattern,   1, sizeof(pattern),   fp);
    fwrite(pattern_2, 1, sizeof(pattern_2), fp);
    fclose(fp);
    printf("权重已保存到文件：%s\n", filename);
}

static void init_model(void) {
    srand((float)(time(NULL) ^ clock()));
    for (size_t i = 0; i < sizeof(pattern); i++)   pattern[i]   = (uint8_t)(rand() & 0xFF);
    for (size_t i = 0; i < sizeof(pattern_2); i++) pattern_2[i] = (uint8_t)(rand() & 0xFF);
}

static int categorical_sample(double* probs) {
    double cum = 0.0;
    double max_val = probs[0];
    for (int i = 1; i < CLASSES; i++) {
        if (probs[i] > max_val) max_val = probs[i];
    }

    double s = 0.0;
    for (int i = 0; i < CLASSES; i++) {
        s += max_val - probs[i];
    }
    s = s / (CLASSES - 1);

    int k = 0;
    int a[CLASSES];
    for (int i = 0; i < CLASSES; i++) {
        if (probs[i] >= s) {
            a[k] = i;
            k++;
        } else {
            probs[i] = 0.0;
        }
    }

    double p = 0.0;
    for (int i = 0; i < k; i++) {
        p += probs[a[i]];
    }
    p = 1 / p;
    for (int i = 0; i < k; i++) {
        probs[a[i]] = probs[a[i]] * p;
    }
    
    double r = (double)rand() / RAND_MAX;
    cum = 0.0;
    for (int i = 0; i < k; i++) {
        cum += probs[a[i]];
        if (r <= cum) {
            return a[i];
        }
    }
    return a[k - 1];
}

static void clauses(uint8_t *clause_outputs) {
    int c = 0;
    for (int k = 0; k < CLAUSES_OF_MEM_LAYER; k++) {
        int a = 0, b = 0;
        for (int i = 0; i < FEATURES_PER_CLAUSE_OF_MEM_LAYER; i++) {
            if (GET_BIT(pattern, c)) {
                if (GET_BIT(mem, i)) {
                    a++;
                } else {
                    b++;
                }
            }
            if (GET_BIT(pattern, FEATURES_OF_MEM_LAYER + c)) {
                if (GET_BIT(mem, i)) {
                    b++;
                } else {
                    a++;
                }
            }
            c++;
        }
        if (b < abs(a - b)) {
            SET_BIT(clause_outputs, k);
        } else {
            CLEAR_BIT(clause_outputs, k);
        }
    }
}

static void clauses_2(uint8_t *clause_outputs, const uint8_t *features) {
    for (int j = 0; j < BLOCKS_OF_META_LAYER; j++) {
        int d = 0;
        for (int k = 0; k < CLAUSES_PER_META_BLOCK; k++) {
            int a = 0, b = 0;
            int c = j * FEATURES_PER_CLAUSE_OF_META;
            for (int i = 0; i < FEATURES_PER_CLAUSE_OF_META; i++) {
                if (GET_BIT(pattern_2, d)) {
                    if (GET_BIT(features, c)) {
                        a++;
                    } else {
                        b++;
                    }
                }
                if (GET_BIT(pattern_2, FEATURES_OF_META_LAYER + d)) {
                    if (GET_BIT(features, c)) {
                        b++;
                    } else {
                        a++;
                    }
                }
                c++;
                d++;
            }
            if (b < abs(a - b)) {
                SET_BIT(clause_outputs, k);
            } else {
                CLEAR_BIT(clause_outputs, k);
            }
        }
    }
}

static void clauses_3(uint8_t *features, uint8_t* pattern) {
    int c = 0;
    for (int k = 0; k < CLAUSES_OF_INPUT_LAYER; k++) {
        int a = 0, b = 0;
        for (int i = 0; i < FEATURES_PER_CLAUSE_OF_INPUT_LAYER; i++) {
            if (GET_BIT(pattern, c)) {
                if (GET_BIT(features, i)) {
                    a++;
                } else {
                    b++;
                }
            }
            if (GET_BIT(pattern, FEATURES_OF_INPUT_LAYER + c)) {
                if (GET_BIT(features, i)) {
                    b++;
                } else {
                    a++;
                }
            }
            c++;
        }
        if (b < abs(a - b)) {
            SET_BIT(mem, k);
        } else {
            CLEAR_BIT(mem, k);
        }
    }
}

static void clauses_4(uint8_t *clause_outputs) {
    int c = 0;
    for (int k = 0; k < CLAUSES_OF_OUT_LAYER; k++) {
        int a = 0, b = 0;
        for (int i = 0; i < FEATURES_PER_CLAUSE_OF_OUT; i++) {
            if (GET_BIT(pattern_3, c)) {
                if (GET_BIT(mem, i)) {
                    a++;
                } else {
                    b++;
                }
            }
            if (GET_BIT(pattern_3, FEATURES_OF_OUT_LAYER + c)) {
                if (GET_BIT(mem, i)) {
                    b++;
                } else {
                    a++;
                }
            }
            c++;
        }
        if (b < abs(a - b)) {
            SET_BIT(clause_outputs, k);
        } else {
            CLEAR_BIT(clause_outputs, k);
        }
    }
}

static void clauses_5(uint8_t *clause_outputs, const uint8_t *features,
                    uint8_t *pattern_ptr, size_t clause_size, size_t feature_size) {

    for (size_t k = 0; k < clause_size; k++) {
        int a = 0, b = 0;
        for (size_t i = 0; i < feature_size; i++) {
            if (GET_BIT(pattern_ptr, k * feature_size + i)) {
                if (GET_BIT(features, i)) {
                    a++;
                } else {
                    b++;
                }
            }
        }
        if (b < abs(a - b)) {
            SET_BIT(clause_outputs, k);
        } else {
            CLEAR_BIT(clause_outputs, k);
        }
    }
}

static bool up(uint8_t* layer, uint8_t* pattern, int features, int clauses, bool r) {
    for (int f = 0; f < clauses; f++) {
        for (int i = 0; i < features; i++) {
            if ((GET_BIT(layer, f * features + i) != GET_BIT(pattern, f * features + i)) == r) {
                if (p >= (float)rand() / RAND_MAX) {
                    TOGGLE_BIT(pattern, f * features + i);
                    return 1;
                }
                return 0;
            }
        }
    }
}

static float train(uint8_t token, uint8_t next_token) {
    srand((float)(time(NULL) ^ clock()));
    uint8_t *features = &token;
    uint8_t mem_layer_output[(CLAUSES_OF_MEM_LAYER + 7) / 8];
    uint8_t meta_layer_output[(CLAUSES_OF_META_LAYER + 7) / 8];
    uint8_t out_layer_outputs[(CLAUSES_OF_OUT_LAYER + 7) / 8];
    uint8_t class_layer_outputs[(CLAUSES_OF_CLASS_LAYER + 7) / 8];
    
    clauses(mem_layer_output);
    clauses_2(meta_layer_output, mem_layer_output);
    clauses_3(features, meta_layer_output);
    clauses_4(out_layer_outputs);

    int logits[CLASSES] = {0};
    
    for (int k = 0; k < CLASSES; k++) {
        int a = 0;
        for (int i = 0; i < CLAUSES_PER_CLASS; i++) {
            if (GET_BIT(class_layer_outputs, k * CLAUSES_PER_CLASS + i)) a++;
        }
        logits[k] = a;
    }

    double probs[CLASSES];
    double max_val = logits[0];
    for (int i = 1; i < CLASSES; i++) {
        if (logits[i] > max_val) max_val = logits[i];
    }
    
    double sum = 0.0;
    for (int j = 0; j < CLASSES; j++) {
        probs[j] = exp((double)(logits[j] - max_val));
        sum += probs[j];
    }
    
    for (int u = 0; u < CLASSES; u++) {
        probs[u] /= sum;
    }

    int idx = categorical_sample(probs);
    for (int i = 0; i < CLASSES; i++) {
        if (next_token == i) {
            float p = 1.0f - (float)probs[i];
            for (int f = 0; f < CLAUSES_PER_CLASS; f++) {
                if(GET_BIT(class_layer_outputs, i * CLAUSES_PER_CLASS + f) {
                    up(mem, pattern_2, FEATURES_PER_CLAUSE_OF_CLASS, CLAUSES_PER_CLASS * CLASSES, 1);
                }
            }
        } else {
            if (probs[i] == 0.0) {
                a;
            } else {
                a;
            }
        }
    }
    if (next_token != idx) {
        

        for (int f = 0; f < CLAUSES_PER_CLASS; f++) {
            if (!GET_BIT(class_layer_outputs, next_token * CLAUSES_PER_CLASS + f)) {
                for (int i = 0; i < FEATURES_PER_CLAUSE_OF_CLASS; i++) {
                    if (GET_BIT(pattern_2, (next_token * CLAUSES_PER_CLASS + f) * FEATURES_PER_CLAUSE_OF_CLASS + i) != GET_BIT(mem, i)) {
                            if (p >= (float)rand() / RAND_MAX) {
                                TOGGLE_BIT(pattern_2, (next_token * CLAUSES_PER_CLASS + f) * FEATURES_PER_CLAUSE_OF_CLASS + i);
                            } else {
                                for (int k = 0; k < FEATURES_PER_CLAUSE_OF_INPUT_LAYER; k++) {
                                    if (GET_BIT(meta_layer_output, i * FEATURES_PER_CLAUSE_OF_INPUT_LAYER + k)) {
                                        if (!GET_BIT(features, k)) {
                                            for (int h = 0; h < FEATURES_PER_CLAUSE_OF_BLOCK; h++) {
                                                if (GET_BIT(pattern, (i * FEATURES_PER_CLAUSE_OF_INPUT_LAYER + k) * FEATURES_PER_CLAUSE_OF_BLOCK + h) == GET_BIT(mem, h)) {
                                                    if (p >= (float)rand() / RAND_MAX) {
                                                        TOGGLE_BIT(pattern, (i * FEATURES_PER_CLAUSE_OF_INPUT_LAYER + k) * FEATURES_PER_CLAUSE_OF_BLOCK + h);
                                                    }
                                                }
                                                if (GET_BIT(pattern, (i * FEATURES_PER_CLAUSE_OF_INPUT_LAYER + k) * FEATURES_PER_CLAUSE_OF_BLOCK + FEATURES_PER_CLAUSE_OF_BLOCK * CLAUSES_OF_META_LAYER + h) != GET_BIT(mem, h)) {
                                                    if (p >= (float)rand() / RAND_MAX) {
                                                        TOGGLE_BIT(pattern, (i * FEATURES_PER_CLAUSE_OF_INPUT_LAYER + k) * FEATURES_PER_CLAUSE_OF_BLOCK + FEATURES_PER_CLAUSE_OF_BLOCK * CLAUSES_OF_META_LAYER + h);
                                                    }
                                                }
                                            }
                                        }
                                    } else {
                                        if (GET_BIT(features, k)) {
                                            for (int h = 0; h < FEATURES_PER_CLAUSE_OF_BLOCK; h++) {
                                                if (GET_BIT(pattern, (i * FEATURES_PER_CLAUSE_OF_INPUT_LAYER + k) * FEATURES_PER_CLAUSE_OF_BLOCK + h) != GET_BIT(mem, h)) {
                                                    if (p >= (float)rand() / RAND_MAX) {
                                                        TOGGLE_BIT(pattern, (i * FEATURES_PER_CLAUSE_OF_INPUT_LAYER + k) * FEATURES_PER_CLAUSE_OF_BLOCK + h);
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    if (GET_BIT(meta_layer_output, i * FEATURES_PER_CLAUSE_OF_INPUT_LAYER + CLAUSES_OF_INPUT_LAYER * FEATURES_PER_CLAUSE_OF_INPUT_LAYER + k)) {
                                        if (GET_BIT(features, k)) {
                                            for (int h = 0; h < FEATURES_PER_CLAUSE_OF_BLOCK; h++) {
                                                if (GET_BIT(pattern, (i * FEATURES_PER_CLAUSE_OF_INPUT_LAYER + CLAUSES_OF_INPUT_LAYER * FEATURES_PER_CLAUSE_OF_INPUT_LAYER + k) * FEATURES_PER_CLAUSE_OF_BLOCK + h) == GET_BIT(mem, h)) {
                                                    if (p >= (float)rand() / RAND_MAX) {
                                                        TOGGLE_BIT(pattern, (i * FEATURES_PER_CLAUSE_OF_INPUT_LAYER + CLAUSES_OF_INPUT_LAYER * FEATURES_PER_CLAUSE_OF_INPUT_LAYER + k) * FEATURES_PER_CLAUSE_OF_BLOCK + h);
                                                    }
                                                }
                                                if (GET_BIT(pattern, (i * FEATURES_PER_CLAUSE_OF_INPUT_LAYER + CLAUSES_OF_INPUT_LAYER * FEATURES_PER_CLAUSE_OF_INPUT_LAYER + k) * FEATURES_PER_CLAUSE_OF_BLOCK + CLAUSES_OF_META_LAYER * FEATURES_PER_CLAUSE_OF_BLOCK + h) != GET_BIT(mem, h)) {
                                                    if (p >= (float)rand() / RAND_MAX) {
                                                        TOGGLE_BIT(pattern, (i * FEATURES_PER_CLAUSE_OF_INPUT_LAYER + CLAUSES_OF_INPUT_LAYER * FEATURES_PER_CLAUSE_OF_INPUT_LAYER + k) * FEATURES_PER_CLAUSE_OF_BLOCK + CLAUSES_OF_META_LAYER * FEATURES_PER_CLAUSE_OF_BLOCK + h);
                                                    }
                                                }
                                            }
                                        }
                                    } else {
                                        if (!GET_BIT(features, k)) {
                                            for (int h = 0; h < FEATURES_PER_CLAUSE_OF_BLOCK; h++) {
                                                if (GET_BIT(pattern, (i * FEATURES_PER_CLAUSE_OF_INPUT_LAYER + CLAUSES_OF_INPUT_LAYER * FEATURES_PER_CLAUSE_OF_INPUT_LAYER + k) * FEATURES_PER_CLAUSE_OF_BLOCK + h) == GET_BIT(mem, h)) {
                                                    if (p >= (float)rand() / RAND_MAX) {
                                                        TOGGLE_BIT(pattern, (i * FEATURES_PER_CLAUSE_OF_INPUT_LAYER + CLAUSES_OF_INPUT_LAYER * FEATURES_PER_CLAUSE_OF_INPUT_LAYER + k) * FEATURES_PER_CLAUSE_OF_BLOCK + h);
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return probs[next_token];
}

int main(int argc, char **argv) {
    char *train_file = NULL;
    char *seed_text = NULL;
    int epochs = 5;
    int gen_tokens = 10;

    if (1 < argc) {
        int i = 1;
        if (strcmp(argv[i], "--train") == 0 || strcmp(argv[i], "-t") == 0) {
            train_file = argv[++i];
        } else if (strcmp(argv[i], "--Generate") == 0 || strcmp(argv[i], "-g") == 0) {
            seed_text = argv[++i];
        }
    }

    FILE *fp = fopen(WEIGHTS_FILE, "rb");
    if (!fp) {
        fp = fopen(WEIGHTS_FILE, "wb");
        if (fp == NULL) {
            perror("创建文件失败");
            return -1;
        }
        init_model();
        fwrite(pattern,   1, sizeof(pattern),   fp);
        fwrite(pattern_2,   1, sizeof(pattern_2),   fp);
        fclose(fp);
    }else {
        fread(pattern,   1, sizeof(pattern),   fp);
        fread(pattern_2, 1, sizeof(pattern_2), fp);
        fclose(fp);
    }
    
    fp = fopen(MEM_FILE, "rb");
    if (!fp) {
        fp = fopen(MEM_FILE, "wb");
        if (fp == NULL) {
            perror("创建文件失败");
            return -1;
        }
        for (size_t i = 0; i < sizeof(mem); i++) mem[i] = 0;
        fwrite(mem, 1, sizeof(mem), fp);
        fclose(fp);
    } else {
        fread(mem, 1, sizeof(mem), fp);
        fclose(fp);
    }
    
    if (train_file) {
        FILE *f = fopen(train_file, "rb");
        if (!f) {
            perror("Error opening training file");
            return 1;
        }

        fseek(f, 0, SEEK_END);
        long len = ftell(f);
        fseek(f, 0, SEEK_SET);

        uint8_t *text = (uint8_t *)malloc(len + 1);
        if (!text) {
            fclose(f);
            perror("malloc text failed");
            return 1;
        }
        fread(text, 1, len, f);
        fclose(f);
        text[len] = '\0';

        printf("Training on %ld characters...\n\n", len);

        for (int epoch = 0; epoch < epochs; epoch++) {
            float total_prob = 0.0f;
            float correct;

            for (long i = 0; i < len - 1; i++) {
                uint8_t token = text[i];
                uint8_t next_token = text[i + 1];

                correct = train(token, next_token);
                total_prob += correct;
                printf("correct = %.4f\n", correct);

            }

            float avg_prob = total_prob / (len - 1);

            printf("Epoch %2d: avg_prob = %.4f\n",
                   epoch + 1, avg_prob);

            if (avg_prob > 0.9f) {
                printf("Early stopping: avg_loss > 0.9\n");
                break;
            }
        }

        free(text);
        printf("\nTraining complete.\n");

        fp = fopen(MEM_FILE, "wb");
        if (fp == NULL) {
            perror("保存记忆文件失败");
            return -1;
        }
        fwrite(mem, 1, sizeof(mem), fp);
        fclose(fp);
        save_weights(WEIGHTS_FILE);
    }
    if (seed_text) {
        printf("\n=== 生成模式 ===\n");
        printf("Seed: %s\n", seed_text);
        printf("Generated: %s", seed_text);   // 先输出 seed

        size_t seed_len = strlen(seed_text);
        uint8_t current = 0;

        /* 先用 seed 更新记忆状态 */
        for (size_t j = 0; j < seed_len; j++) {
            current = (uint8_t)seed_text[j];
            forward(current);                  // 更新 mem
        }

        /* 生成新 token */
        for (int i = 0; i < gen_tokens; i++) {
            current = forward(current);
            putchar(current);
        }
        putchar('\n');
        printf("生成完成（%d 个新 token）\n", gen_tokens);
    }

    fp = fopen(MEM_FILE, "wb");
    if (fp == NULL) {
        perror("保存记忆文件失败");
        return -1;
    }
    fwrite(mem, 1, sizeof(mem), fp);
    fclose(fp);

    return 0;
}
