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

#define OUT_PATTERN_BYTES_PER_CLASS ((OUT_CLAUSES * MEM * 2 + 7) / 8)

#define BIT_ARRAY_SIZE 100
uint8_t bit_array[(BIT_ARRAY_SIZE + 7) / 8];

#define SET_BIT(arr, n)     ((arr)[(n)/8] |=  (1U << ((n)%8)))
#define CLEAR_BIT(arr, n)   ((arr)[(n)/8] &= ~(1U << ((n)%8)))
#define GET_BIT(arr, n)     (((arr)[(n)/8] & (1U << ((n)%8))) != 0)
#define TOGGLE_BIT(arr, n)  ((arr)[(n)/8] ^= (1U << ((n)%8)))

/* ==================== 全局模式数组（模型权重） ==================== */
uint8_t pattern[(CLAUSES * CLAUSE_FEATURES * 2 + 7) / 8];           
uint8_t pattern_2[(META_CLAUSES * MEM * 2 + 7) / 8];               
uint8_t pattern_3[VOCAB_SIZE * OUT_PATTERN_BYTES_PER_CLASS];       
uint8_t hyper_pattern[(MEM * CLAUSES * 2 + 7) / 8];                
uint8_t mem[(MEM + 7) / 8] = {0};                                  

static const char *DEFAULT_WEIGHTS = "tmlm.weights";

static void save_weights(const char *filename) {
    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        perror("保存权重文件失败");
        return;
    }
    fwrite(pattern,       1, sizeof(pattern),       fp);
    fwrite(pattern_2,     1, sizeof(pattern_2),     fp);
    fwrite(pattern_3,     1, sizeof(pattern_3),     fp);
    fwrite(hyper_pattern, 1, sizeof(hyper_pattern), fp);
    fclose(fp);
    printf("权重已保存到文件：%s\n", filename);
}

static int load_weights(const char *filename) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        printf("权重文件 %s 不存在，将随机初始化模型...\n", filename);
        return 0;
    }
    size_t r1 = fread(pattern,       1, sizeof(pattern),       fp);
    size_t r2 = fread(pattern_2,     1, sizeof(pattern_2),     fp);
    size_t r3 = fread(pattern_3,     1, sizeof(pattern_3),     fp);
    size_t r4 = fread(hyper_pattern, 1, sizeof(hyper_pattern), fp);
    fclose(fp);

    if (r1 == sizeof(pattern) && r2 == sizeof(pattern_2) &&
        r3 == sizeof(pattern_3) && r4 == sizeof(hyper_pattern)) {
        printf("权重已从文件 %s 成功加载\n", filename);
        return 1;
    } else {
        printf("权重文件 %s 损坏或大小不匹配，将随机初始化...\n", filename);
        return 0;
    }
}

/* ==================== 初始化模型（仅在无权重文件时调用） ==================== */
static void init_model(void) {
    srand(12345);

    for (size_t i = 0; i < sizeof(pattern); i++)       pattern[i]       = (uint8_t)(rand() & 0xFF);
    for (size_t i = 0; i < sizeof(pattern_2); i++)     pattern_2[i]     = (uint8_t)(rand() & 0xFF);
    for (size_t i = 0; i < sizeof(pattern_3); i++)     pattern_3[i]     = (uint8_t)(rand() & 0xFF);
    for (size_t i = 0; i < sizeof(hyper_pattern); i++) hyper_pattern[i] = (uint8_t)(rand() & 0xFF);
}

static int categorical_sample(const double* probs, int num_classes) {
    double cum = 0.0;
    double max_val = probs[0];
    for (int i = 1; i < num_classes; i++) {
        if (probs[i] > max_val) max_val = probs[i];
    }

    double s = 0.0;
    for (int i = 0; i < num_classes; i++) {
        s += max_val - probs[i];
    }
    s = s / num_classes;

    int k = 0;
    int a[256];
    for (int i = 0; i < num_classes; i++) {
        if (probs[i] > s) {
            a[k++] = i;
        }
    }

    if (k == 0) return 0;

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
    memset(clause_outputs, 0, (clause_size + 7) / 8);

    for (size_t k = 0; k < clause_size; k++) {
        int a = 0, b = 0;
        for (size_t i = 0; i < feature_size; i++) {
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
    memset(clause_outputs, 0, (clause_size + 7) / 8);

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
    uint8_t features[1] = {token};

    uint8_t clause_outputs[(CLAUSES + 7) / 8] = {0};
    clauses(clause_outputs, features, pattern, CLAUSES, CLAUSE_FEATURES);

    uint8_t meta_clause_outputs[(META_CLAUSES + 7) / 8] = {0};
    clauses(meta_clause_outputs, mem, pattern_2, META_CLAUSES, MEM);

    uint8_t hyper_clause_outputs[(MEM + 7) / 8] = {0};
    clauses(hyper_clause_outputs, clause_outputs, hyper_pattern, MEM, CLAUSES);

    int logits[VOCAB_SIZE] = {0};
    for (int k = 0; k < VOCAB_SIZE; k++) {
        uint8_t out_clause_outputs[(OUT_CLAUSES + 7) / 8] = {0};
        uint8_t *class_pattern = pattern_3 + (size_t)k * OUT_PATTERN_BYTES_PER_CLASS;
        clauses(out_clause_outputs, hyper_clause_outputs, class_pattern, OUT_CLAUSES, MEM);

        int a = 0;
        for (int i = 0; i < OUT_CLAUSES; i++) {
            if (GET_BIT(out_clause_outputs, i)) a++;
        }
        logits[k] = a;
    }

    double probs[VOCAB_SIZE];
    double max_val = logits[0];
    for (int i = 1; i < VOCAB_SIZE; i++) {
        if (logits[i] > max_val) max_val = logits[i];
    }

    double sum = 0.0;
    for (int i = 0; i < VOCAB_SIZE; i++) {
        probs[i] = exp((double)(logits[i] - max_val));
        sum += probs[i];
    }
    for (int i = 0; i < VOCAB_SIZE; i++) {
        probs[i] /= sum;
    }

    return categorical_sample(probs, VOCAB_SIZE);
}

static int sample_next(uint8_t current) {
    return forward(current);
}

static float train(uint8_t token, uint8_t next_token) {
    uint8_t features[1] = {token};

    uint8_t clause_outputs[(CLAUSES + 7) / 8] = {0};
    clauses(clause_outputs, features, pattern, CLAUSES, CLAUSE_FEATURES);

    uint8_t meta_clause_outputs[(META_CLAUSES + 7) / 8] = {0};
    clauses(meta_clause_outputs, mem, pattern_2, META_CLAUSES, MEM);

    uint8_t hyper_clause_outputs[(MEM + 7) / 8] = {0};
    clauses(hyper_clause_outputs, clause_outputs, hyper_pattern, MEM, CLAUSES);

    size_t out_bytes = (OUT_CLAUSES + 7) / 8;
    uint8_t *out_clause_outputs = (uint8_t *)malloc(VOCAB_SIZE * out_bytes);
    if (out_clause_outputs == NULL) {
        perror("malloc failed in train");
        exit(1);
    }
    memset(out_clause_outputs, 0, VOCAB_SIZE * out_bytes);

    int logits[VOCAB_SIZE] = {0};
    for (int k = 0; k < VOCAB_SIZE; k++) {
        uint8_t *class_out = out_clause_outputs + (size_t)k * out_bytes;
        uint8_t *class_pattern = pattern_3 + (size_t)k * OUT_PATTERN_BYTES_PER_CLASS;
        clauses(class_out, hyper_clause_outputs, class_pattern, OUT_CLAUSES, MEM);

        int a = 0;
        for (int i = 0; i < OUT_CLAUSES; i++) {
            if (GET_BIT(class_out, i)) a++;
        }
        logits[k] = a;
    }

    double probs[VOCAB_SIZE];
    double max_val = logits[0];
    for (int i = 1; i < VOCAB_SIZE; i++) {
        if (logits[i] > max_val) max_val = logits[i];
    }
    double sum = 0.0;
    for (int i = 0; i < VOCAB_SIZE; i++) {
        probs[i] = exp((double)(logits[i] - max_val));
        sum += probs[i];
    }
    for (int i = 0; i < VOCAB_SIZE; i++) probs[i] /= sum;

    int idx = categorical_sample(probs, VOCAB_SIZE);

    if (next_token != idx) {
        uint8_t *correct_pattern = pattern_3 + (size_t)next_token * OUT_PATTERN_BYTES_PER_CLASS;
        for (int f = 0; f < 16; f++) {
            size_t bit_pos = (size_t)rand() % (OUT_CLAUSES * MEM * 2);
            TOGGLE_BIT(correct_pattern, bit_pos);
        }

        uint8_t *wrong_pattern = pattern_3 + (size_t)idx * OUT_PATTERN_BYTES_PER_CLASS;
        for (int f = 0; f < 8; f++) {
            size_t bit_pos = (size_t)rand() % (OUT_CLAUSES * MEM * 2);
            TOGGLE_BIT(wrong_pattern, bit_pos);
        }
    }

    float prob = probs[next_token];
    float loss = -logf(fmaxf(prob, 1e-10f));

    free(out_clause_outputs);
    return loss;
}

/* ==================== 主函数（新增权重文件支持） ==================== */
int main(int argc, char **argv) {
    const char *train_file = NULL;
    const char *seed_text = "a";
    const char *weights_file = DEFAULT_WEIGHTS;
    int epochs = 20;
    int gen_tokens = 50;

    /* 命令行参数解析（新增 --weights / -w） */
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

    /* 权重加载逻辑：文件存在 → 读取；不存在 → 随机初始化 */
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

        /* 训练结束后自动保存权重 */
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
