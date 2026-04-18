#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>

#define VOCAB_SIZE 256
#define MEM 512

#define BLOCKS_OF_META_LAYER MEM
#define CLAUSES_PER_BLOCK VOCAB_SIZE * 2
#define CLAUSES_OF_META_LAYER BLOCKS_OF_META_LAYER * CLAUSES_PER_BLOCK
#define FEATURES_PER_CLAUSE_OF_BLOCK MEM * 2


#define CLAUSES_OF_INPUT_LAYER BLOCKS_OF_META_LAYER
#define FEATURES_PER_CLAUSE_OF_INPUT_LAYER CLAUSES_PER_BLOCK

#define CLASSES VOCAB_SIZE
#define CLAUSES_PER_CLASS 256
#define CLAUSES_OF_CLASS_LAYER CLASSES * CLAUSES_PER_CLASS
#define FEATURES_PER_CLAUSE_OF_CLASS MEM


#define SET_BIT(arr, n)     ((arr)[(n)/8] |=  (1U << ((n)%8)))
#define CLEAR_BIT(arr, n)   ((arr)[(n)/8] &= ~(1U << ((n)%8)))
#define GET_BIT(arr, n)     (((arr)[(n)/8] & (1U << ((n)%8))) != 0)
#define TOGGLE_BIT(arr, n)  ((arr)[(n)/8] ^= (1U << ((n)%8)))

uint8_t pattern[(MEM * MEM * 2 + 7) / 8];
uint8_t pattern_2[(FEATURES_PER_CLAUSE_OF_CLASS * CLAUSES_OF_CLASS_LAYER + 7) / 8];                               

static const char *DEFAULT_WEIGHTS = "tmlm.weights";

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

static int load_weights(const char *filename) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        printf("权重文件 %s 不存在，将随机初始化模型...\n", filename);
        return 0;
    }
    size_t r1 = fread(pattern,   1, sizeof(pattern),   fp);
    size_t r2 = fread(pattern_2, 1, sizeof(pattern_2), fp);
    fclose(fp);

    if (r1 == sizeof(pattern) && r2 == sizeof(pattern_2)) {
        printf("权重已从文件 %s 成功加载\n", filename);
        return 1;
    } else {
        printf("权重文件 %s 损坏或大小不匹配，将随机初始化...\n", filename);
        return 0;
    }
}

static void init_model(void) {
    srand(12345);
    for (size_t i = 0; i < sizeof(pattern); i++)   pattern[i]   = (uint8_t)(rand() & 0xFF);
    for (size_t i = 0; i < sizeof(pattern_2); i++) pattern_2[i] = (uint8_t)(rand() & 0xFF);
}

static int categorical_sample(const double* probs) {
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
        if (probs[i] > s) {
            a[k++] = i;
        }
    }

    double p = 0.0;
    for (int i = 0; i < k; i++) {
        p += probs[a[i]];
    }

    double r = (double)rand() / RAND_MAX * p;
    cum = 0.0;
    for (int i = 0; i < k; i++) {
        cum += probs[a[i]];
        if (r <= cum) {
            return a[i];
        }
    }
    return a[k - 1];
}

static void clauses(uint8_t *clause_outputs, const uint8_t *features,
                    uint8_t *pattern_ptr, size_t clause_size, size_t feature_size) {

    for (int k = 0; k < clause_size; k++) {
        int a = 0, b = 0;
        for (int i = 0; i < feature_size; i++) {
            if (GET_BIT(pattern_ptr, k * feature_size + i)) {
                if (GET_BIT(features, i)) a++;
                else b++;
            }
            if (GET_BIT(pattern_ptr, k * feature_size + i + clause_size * feature_size)) {
                if (GET_BIT(features, i)) b++;
                else a++;
            }
        }
        if (b < abs(a - b)) {
            SET_BIT(clause_outputs, k);
        }
    }
}

static void clauses_2(uint8_t *clause_outputs, const uint8_t *features,
                    uint8_t *pattern_ptr, size_t clause_size, size_t feature_size) {

    for (size_t k = 0; k < clause_size; k++) {
        int a = 0, b = 0;
        for (size_t i = 0; i < feature_size; i++) {
            if (GET_BIT(pattern_ptr, k * feature_size + i)) {
                if (GET_BIT(features, i)) a++;
                else b++;
            }
        }
        if (b < abs(a - b)) {
            SET_BIT(clause_outputs, k);
        }
    }
}

static int forward(uint8_t token) {
    uint8_t *features = &token;

    uint8_t meta_layer_output[(CLAUSES_OF_META_LAYER + 7) / 8] = {0};
    uint8_t input_layer_output[(CLAUSES_OF_INPUT_LAYER + 7) / 8] = {0};

    clauses(meta_layer_output, mem, pattern, CLAUSES_OF_META_LAYER, FEATURES_PER_CLAUSE_OF_BLOCK);
    clauses(input_layer_output, features, meta_layer_output, CLAUSES_OF_INPUT_LAYER, FEATURES_PER_CLAUSE_OF_INPUT_LAYER);

    uint8_t *class_layer_outputs = (uint8_t *)malloc((CLAUSES_OF_CLASS_LAYER + 7) / 8);
    if (class_layer_outputs == NULL) {
        perror("malloc failed in train");
        exit(1);
    }
    memset(class_layer_outputs, 0, (CLAUSES_OF_CLASS_LAYER + 7) / 8);

    int logits[CLASSES] = {0};
    clauses_2(class_layer_outputs, mem, pattern_2, CLAUSES_OF_CLASS_LAYER, FEATURES_PER_CLAUSE_OF_CLASS);
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
    for (int i = 0; i < CLASSES; i++) {
        probs[i] = exp((double)(logits[i] - max_val));
        sum += probs[i];
    }
    for (int i = 0; i < CLASSES; i++) probs[i] /= sum;

    return categorical_sample(probs);
}

static int sample_next(uint8_t current) {
    return forward(current);
}

static float train(uint8_t token, uint8_t next_token) {
    uint8_t *features = &token;

    uint8_t meta_layer_output[(CLAUSES_OF_META_LAYER + 7) / 8] = {0};
    uint8_t input_layer_output[(CLAUSES_OF_INPUT_LAYER + 7) / 8] = {0};

    clauses(meta_layer_output, mem, pattern, CLAUSES_OF_META_LAYER, FEATURES_PER_CLAUSE_OF_BLOCK);
    clauses(input_layer_output, features, meta_layer_output, CLAUSES_OF_INPUT_LAYER, FEATURES_PER_CLAUSE_OF_INPUT_LAYER);

    uint8_t *class_layer_outputs = (uint8_t *)malloc((CLAUSES_OF_CLASS_LAYER + 7) / 8);
    if (class_layer_outputs == NULL) {
        perror("malloc failed in train");
        exit(1);
    }
    memset(class_layer_outputs, 0, (CLAUSES_OF_CLASS_LAYER + 7) / 8);

    int logits[CLASSES] = {0};
    clauses_2(class_layer_outputs, mem, pattern_2, CLAUSES_OF_CLASS_LAYER, FEATURES_PER_CLAUSE_OF_CLASS);
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
    for (int i = 0; i < CLASSES; i++) {
        probs[i] = exp((double)(logits[i] - max_val));
        sum += probs[i];
    }
    for (int i = 0; i < CLASSES; i++) probs[i] /= sum;

    int idx = categorical_sample(probs);

    if (next_token != idx) {
        float p = 1.0f - (float)probs[next_token];

        for (int f = 0; f < CLAUSES_PER_CLASS; f++) {
            if (!GET_BIT(class_layer_outputs, next_token * CLAUSES_PER_CLASS + f)) {
                for (int i = 0; i < FEATURES_PER_CLAUSE_OF_CLASS; i++) {
                    if (GET_BIT(pattern_2, (next_token * CLAUSES_PER_CLASS + f) * FEATURES_PER_CLAUSE_OF_CLASS + i)) {
                        if (!GET_BIT(input_layer_output, i)) {
                            if (p >= (float)rand() / RAND_MAX) {
                                TOGGLE_BIT(pattern_2, (next_token * CLAUSES_PER_CLASS + f) * FEATURES_PER_CLAUSE_OF_CLASS + i);
                            } else {
                                for (int k = 0; k < FEATURES_PER_CLAUSE_OF_INPUT_LAYER; k++) {
                                    if (GET_BIT(meta_layer_output, ((next_token * CLAUSES_PER_CLASS + f) * FEATURES_PER_CLAUSE_OF_CLASS + i) * FEATURES_PER_CLAUSE_OF_INPUT_LAYER + k)) {
                                        if (!GET_BIT(features, k)) {
                                            for (int h = 0; h < FEATURES_PER_CLAUSE_OF_BLOCK; h++) {
                                                if (GET_BIT(pattern, (((next_token * CLAUSES_PER_CLASS + f) * FEATURES_PER_CLAUSE_OF_CLASS + i) * FEATURES_PER_CLAUSE_OF_INPUT_LAYER + k) * FEATURES_PER_CLAUSE_OF_BLOCK + h) == GET_BIT(mem, h)) {
                                                    if (p >= (float)rand() / RAND_MAX) {
                                                        TOGGLE_BIT(pattern, (((next_token * CLAUSES_PER_CLASS + f) * FEATURES_PER_CLAUSE_OF_CLASS + i) * FEATURES_PER_CLAUSE_OF_INPUT_LAYER + k) * FEATURES_PER_CLAUSE_OF_BLOCK + h);
                                                    }
                                                }
                                            }
                                        }
                                    } else {
                                        if (GET_BIT(features, k)) {
                                            for (int h = 0; h < FEATURES_PER_CLAUSE_OF_BLOCK; h++) {
                                                if (GET_BIT(pattern, (((next_token * CLAUSES_PER_CLASS + f) * FEATURES_PER_CLAUSE_OF_CLASS + i) * FEATURES_PER_CLAUSE_OF_INPUT_LAYER + k) * FEATURES_PER_CLAUSE_OF_BLOCK + h) != GET_BIT(mem, h)) {
                                                    if (p >= (float)rand() / RAND_MAX) {
                                                        TOGGLE_BIT(pattern, (((next_token * CLAUSES_PER_CLASS + f) * FEATURES_PER_CLAUSE_OF_CLASS + i) * FEATURES_PER_CLAUSE_OF_INPUT_LAYER + k) * FEATURES_PER_CLAUSE_OF_BLOCK + h);
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        if (GET_BIT(input_layer_output, i)) {
                            if (p >= (float)rand() / RAND_MAX) {
                                TOGGLE_BIT(pattern_2, (next_token * CLAUSES_PER_CLASS + f) * FEATURES_PER_CLAUSE_OF_CLASS + i);
                            }
                        }
                    }
                }
            }
        }
    }

    free(out_clause_outputs);
    return probs[next_token];
}

/* ==================== 主函数 ==================== */
int main(int argc, char **argv) {
    const char *train_file = NULL;
    const char *seed_text = "a";
    const char *weights_file = DEFAULT_WEIGHTS;
    int epochs = 20;
    int gen_tokens = 50;

    /* 命令行参数解析 */
    for (int i = 1; i < argc; i++) {
        if (i + 1 < argc) {
            if (strcmp(argv[i], "--train") == 0 || strcmp(argv[i], "-t") == 0) {
                train_file = argv[++i];
            } else if (strcmp(argv[i], "--seed") == 0 || strcmp(argv[i], "-s") == 0) {
                seed_text = argv[++i];
            } else if (strcmp(argv[i], "--tokens") == 0 || strcmp(argv[i], "-n") == 0) {
                gen_tokens = atoi(argv[++i]);
            } else if (strcmp(argv[i], "--epochs") == 0 || strcmp(argv[i], "-e") == 0) {
                epochs = atoi(argv[++i]);
            } else if (strcmp(argv[i], "--weights") == 0 || strcmp(argv[i], "-w") == 0) {
                weights_file = argv[++i];
            }
        }
    }

    /* 加载权重或随机初始化 */
    int loaded = load_weights(weights_file);
    if (!loaded) {
        init_model();
    }

    /* ==================== 训练阶段 ==================== */
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
            float total_loss = 0.0f;
            int correct = 0;

            for (long i = 0; i < len - 1; i++) {
                uint8_t token = text[i];
                uint8_t next_token = text[i + 1];

                total_loss += train(token, next_token);

                int predicted = sample_next(token);
                if (predicted == next_token) correct++;
            }

            float avg_loss = total_loss / (len - 1);
            float accuracy = (float)correct / (len - 1) * 100.0f;

            printf("Epoch %2d: loss = %.4f | accuracy = %.2f%%\n",
                   epoch + 1, avg_loss, accuracy);

            if (accuracy > 95.0f) {
                printf("Early stopping: accuracy > 95%%\n");
                break;
            }
        }

        free(text);
        printf("\nTraining complete.\n");

        save_weights(weights_file);
    }

    /* ==================== 生成阶段 ==================== */
    printf("\nGenerating text with seed: \"%s\"\n\n", seed_text);
    printf("%s", seed_text);

    if (strlen(seed_text) == 0) {
        printf("\n");
        return 0;
    }

    uint8_t current = (uint8_t)seed_text[strlen(seed_text) - 1];
    for (int i = 0; i < gen_tokens; i++) {
        int next = sample_next(current);
        printf("%c", (char)next);
        current = (uint8_t)next;
    }
    printf("\n");

    return 0;
}
