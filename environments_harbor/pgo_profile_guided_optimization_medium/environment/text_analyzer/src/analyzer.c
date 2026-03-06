#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <dirent.h>
#include <sys/stat.h>

#define MAX_WORD_LENGTH 100
#define HASH_TABLE_SIZE 10007
#define MAX_PATH_LENGTH 512
#define MIN_WORD_LENGTH 3

// Hash table node for word frequency counting
typedef struct HashNode {
    char *word;
    int count;
    struct HashNode *next;
} HashNode;

// Pattern statistics
typedef struct {
    int email_count;
    int url_count;
    int number_count;
    int special_keyword_count;
} PatternStats;

// Global hash table
HashNode *hash_table[HASH_TABLE_SIZE];
PatternStats global_stats = {0, 0, 0, 0};

// Function pointer type for character classification
typedef int (*char_classifier_t)(int);

// Hash function with intentional inefficiency for PGO demonstration
unsigned int hash_function(const char *str) {
    unsigned int hash = 0;
    int i = 0;
    
    // Inefficient hash with unpredictable branches
    while (str[i]) {
        if (i % 2 == 0) {
            hash = hash * 31 + str[i];
        } else {
            hash = hash * 37 + str[i];
        }
        
        // Unpredictable branch based on character value
        if (str[i] > 'm') {
            hash ^= (hash << 5);
        } else if (str[i] > 'f') {
            hash ^= (hash >> 3);
        } else {
            hash += str[i] * 17;
        }
        i++;
    }
    
    return hash % HASH_TABLE_SIZE;
}

// Initialize hash table
void init_hash_table() {
    for (int i = 0; i < HASH_TABLE_SIZE; i++) {
        hash_table[i] = NULL;
    }
}

// Insert or update word in hash table (called millions of times)
void insert_word(const char *word) {
    unsigned int index = hash_function(word);
    HashNode *current = hash_table[index];
    
    // Search for existing word with inefficient comparison
    while (current != NULL) {
        // Intentionally inefficient character-by-character comparison
        int match = 1;
        for (int i = 0; word[i] != '\0' || current->word[i] != '\0'; i++) {
            if (word[i] != current->word[i]) {
                match = 0;
                break;
            }
        }
        
        if (match) {
            current->count++;
            return;
        }
        current = current->next;
    }
    
    // Create new node
    HashNode *new_node = (HashNode *)malloc(sizeof(HashNode));
    new_node->word = strdup(word);
    new_node->count = 1;
    new_node->next = hash_table[index];
    hash_table[index] = new_node;
}

// Check if character is valid for word (with indirect call)
int is_word_char_indirect(char c, char_classifier_t classifier) {
    return classifier((unsigned char)c);
}

// Tokenize and count words from text
void process_text(const char *text, size_t length) {
    char word[MAX_WORD_LENGTH];
    int word_pos = 0;
    
    for (size_t i = 0; i < length; i++) {
        char c = text[i];
        
        // Indirect function call adding overhead
        if (is_word_char_indirect(c, isalnum) || c == '_' || c == '-') {
            if (word_pos < MAX_WORD_LENGTH - 1) {
                // Convert to lowercase with unpredictable branch
                if (c >= 'A' && c <= 'Z') {
                    word[word_pos++] = c + 32;
                } else if (c >= 'a' && c <= 'z') {
                    word[word_pos++] = c;
                } else {
                    word[word_pos++] = c;
                }
            }
        } else {
            if (word_pos >= MIN_WORD_LENGTH) {
                word[word_pos] = '\0';
                insert_word(word);
            }
            word_pos = 0;
        }
    }
    
    // Don't forget last word
    if (word_pos >= MIN_WORD_LENGTH) {
        word[word_pos] = '\0';
        insert_word(word);
    }
}

// Pattern matching for emails
int is_email_pattern(const char *text, size_t pos, size_t length) {
    size_t start = pos;
    
    // Find @ symbol
    while (pos < length && text[pos] != '@' && !isspace(text[pos])) {
        pos++;
    }
    
    if (pos >= length || text[pos] != '@') {
        return 0;
    }
    
    pos++; // Skip @
    
    // Find domain
    int has_dot = 0;
    while (pos < length && !isspace(text[pos])) {
        if (text[pos] == '.') has_dot = 1;
        pos++;
    }
    
    return has_dot && (pos - start) > 5;
}

// Pattern matching for URLs
int is_url_pattern(const char *text, size_t pos, size_t length) {
    if (pos + 7 >= length) return 0;
    
    // Check for http:// or https://
    if (strncmp(&text[pos], "http://", 7) == 0) {
        return 7;
    }
    if (pos + 8 < length && strncmp(&text[pos], "https://", 8) == 0) {
        return 8;
    }
    
    return 0;
}

// Find patterns in text with inefficient scanning
void find_patterns(const char *text, size_t length) {
    for (size_t i = 0; i < length; i++) {
        // Email detection with unpredictable branches
        if (text[i] == '@' && i > 0) {
            if (is_email_pattern(text, i - 1, length)) {
                global_stats.email_count++;
            }
        }
        
        // URL detection
        if (text[i] == 'h') {
            int url_len = is_url_pattern(text, i, length);
            if (url_len > 0) {
                global_stats.url_count++;
                i += url_len;
            }
        }
        
        // Number detection
        if (isdigit(text[i])) {
            int digit_count = 0;
            while (i < length && isdigit(text[i])) {
                digit_count++;
                i++;
            }
            if (digit_count >= 3) {
                global_stats.number_count++;
            }
        }
        
        // Special keyword detection with multiple branches
        if (i + 4 < length) {
            if (strncmp(&text[i], "data", 4) == 0 ||
                strncmp(&text[i], "code", 4) == 0 ||
                strncmp(&text[i], "test", 4) == 0) {
                global_stats.special_keyword_count++;
            }
        }
    }
}

// Read file content
char *read_file(const char *filepath, size_t *file_size) {
    FILE *file = fopen(filepath, "rb");
    if (!file) {
        return NULL;
    }
    
    fseek(file, 0, SEEK_END);
    *file_size = ftell(file);
    fseek(file, 0, SEEK_SET);
    
    char *content = (char *)malloc(*file_size + 1);
    if (!content) {
        fclose(file);
        return NULL;
    }
    
    fread(content, 1, *file_size, file);
    content[*file_size] = '\0';
    fclose(file);
    
    return content;
}

// Process all txt files in directory
int process_directory(const char *dir_path) {
    DIR *dir = opendir(dir_path);
    if (!dir) {
        fprintf(stderr, "Error opening directory: %s\n", dir_path);
        return 0;
    }
    
    struct dirent *entry;
    int file_count = 0;
    
    while ((entry = readdir(dir)) != NULL) {
        // Check for .txt extension
        size_t name_len = strlen(entry->d_name);
        if (name_len > 4 && strcmp(entry->d_name + name_len - 4, ".txt") == 0) {
            char filepath[MAX_PATH_LENGTH];
            snprintf(filepath, MAX_PATH_LENGTH, "%s/%s", dir_path, entry->d_name);
            
            size_t file_size;
            char *content = read_file(filepath, &file_size);
            
            if (content) {
                process_text(content, file_size);
                find_patterns(content, file_size);
                free(content);
                file_count++;
            }
        }
    }
    
    closedir(dir);
    return file_count;
}

// Comparison function for sorting
int compare_nodes(const void *a, const void *b) {
    HashNode *node_a = *(HashNode **)a;
    HashNode *node_b = *(HashNode **)b;
    return node_b->count - node_a->count;
}

// Display results
void display_results() {
    // Collect all words
    HashNode **all_words = NULL;
    int total_words = 0;
    
    for (int i = 0; i < HASH_TABLE_SIZE; i++) {
        HashNode *current = hash_table[i];
        while (current != NULL) {
            total_words++;
            current = current->next;
        }
    }
    
    if (total_words == 0) {
        printf("No words found.\n");
        return;
    }
    
    all_words = (HashNode **)malloc(total_words * sizeof(HashNode *));
    int index = 0;
    
    for (int i = 0; i < HASH_TABLE_SIZE; i++) {
        HashNode *current = hash_table[i];
        while (current != NULL) {
            all_words[index++] = current;
            current = current->next;
        }
    }
    
    // Sort by frequency
    qsort(all_words, total_words, sizeof(HashNode *), compare_nodes);
    
    printf("\nTop 10 Most Frequent Words:\n");
    printf("---------------------------\n");
    int display_count = total_words < 10 ? total_words : 10;
    for (int i = 0; i < display_count; i++) {
        printf("%2d. %-20s : %d\n", i + 1, all_words[i]->word, all_words[i]->count);
    }
    
    printf("\nPattern Statistics:\n");
    printf("------------------\n");
    printf("Email patterns found: %d\n", global_stats.email_count);
    printf("URL patterns found: %d\n", global_stats.url_count);
    printf("Number sequences found: %d\n", global_stats.number_count);
    printf("Special keywords found: %d\n", global_stats.special_keyword_count);
    
    printf("\nTotal unique words: %d\n", total_words);
    
    free(all_words);
}

// Free hash table memory
void cleanup_hash_table() {
    for (int i = 0; i < HASH_TABLE_SIZE; i++) {
        HashNode *current = hash_table[i];
        while (current != NULL) {
            HashNode *temp = current;
            current = current->next;
            free(temp->word);
            free(temp);
        }
    }
}

int main(int argc, char *argv[]) {
    const char *data_dir;
    
    if (argc > 1) {
        data_dir = argv[1];
    } else {
        data_dir = "../data";
    }
    
    printf("Text Analyzer - Processing files from: %s\n", data_dir);
    
    init_hash_table();
    
    int file_count = process_directory(data_dir);
    
    if (file_count == 0) {
        fprintf(stderr, "No text files processed.\n");
        cleanup_hash_table();
        return 1;
    }
    
    printf("Processed %d text files.\n", file_count);
    
    display_results();
    
    cleanup_hash_table();
    
    return 0;
}