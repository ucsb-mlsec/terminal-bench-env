#include "json_parser.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

typedef struct {
    const char *data;
    size_t len;
    size_t pos;
} parser_state_t;

static void skip_whitespace(parser_state_t *state) {
    while (state->pos < state->len) {
        char c = state->data[state->pos];
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
            state->pos++;
        } else {
            break;
        }
    }
}

static int parse_value(parser_state_t *state);

static int parse_string(parser_state_t *state) {
    if (state->pos >= state->len || state->data[state->pos] != '"') {
        return -1;
    }
    state->pos++;
    
    while (state->pos < state->len) {
        char c = state->data[state->pos];
        if (c == '"') {
            state->pos++;
            return 0;
        } else if (c == '\\') {
            state->pos++;
            if (state->pos >= state->len) {
                return -1;
            }
            state->pos++;
        } else {
            state->pos++;
        }
    }
    return -1;
}

static int parse_number(parser_state_t *state) {
    if (state->pos >= state->len) {
        return -1;
    }
    
    if (state->data[state->pos] == '-') {
        state->pos++;
    }
    
    if (state->pos >= state->len) {
        return -1;
    }
    
    if (state->data[state->pos] < '0' || state->data[state->pos] > '9') {
        return -1;
    }
    
    while (state->pos < state->len && state->data[state->pos] >= '0' && state->data[state->pos] <= '9') {
        state->pos++;
    }
    
    if (state->pos < state->len && state->data[state->pos] == '.') {
        state->pos++;
        while (state->pos < state->len && state->data[state->pos] >= '0' && state->data[state->pos] <= '9') {
            state->pos++;
        }
    }
    
    return 0;
}

static int parse_keyword(parser_state_t *state, const char *keyword) {
    size_t kw_len = strlen(keyword);
    if (state->pos + kw_len > state->len) {
        return -1;
    }
    
    if (memcmp(state->data + state->pos, keyword, kw_len) == 0) {
        state->pos += kw_len;
        return 0;
    }
    return -1;
}

static int parse_array(parser_state_t *state) {
    if (state->pos >= state->len || state->data[state->pos] != '[') {
        return -1;
    }
    state->pos++;
    
    skip_whitespace(state);
    
    if (state->pos < state->len && state->data[state->pos] == ']') {
        state->pos++;
        return 0;
    }
    
    while (state->pos < state->len) {
        if (parse_value(state) != 0) {
            return -1;
        }
        
        skip_whitespace(state);
        
        if (state->pos >= state->len) {
            return -1;
        }
        
        if (state->data[state->pos] == ']') {
            state->pos++;
            return 0;
        } else if (state->data[state->pos] == ',') {
            state->pos++;
            skip_whitespace(state);
        } else {
            return -1;
        }
    }
    
    return -1;
}

static int parse_object(parser_state_t *state) {
    if (state->pos >= state->len || state->data[state->pos] != '{') {
        return -1;
    }
    state->pos++;
    
    skip_whitespace(state);
    
    if (state->pos < state->len && state->data[state->pos] == '}') {
        state->pos++;
        return 0;
    }
    
    while (state->pos < state->len) {
        skip_whitespace(state);
        
        if (parse_string(state) != 0) {
            return -1;
        }
        
        skip_whitespace(state);
        
        if (state->pos >= state->len || state->data[state->pos] != ':') {
            return -1;
        }
        state->pos++;
        
        skip_whitespace(state);
        
        if (parse_value(state) != 0) {
            return -1;
        }
        
        skip_whitespace(state);
        
        if (state->pos >= state->len) {
            return -1;
        }
        
        if (state->data[state->pos] == '}') {
            state->pos++;
            return 0;
        } else if (state->data[state->pos] == ',') {
            state->pos++;
        } else {
            return -1;
        }
    }
    
    return -1;
}

static int parse_value(parser_state_t *state) {
    skip_whitespace(state);
    
    if (state->pos >= state->len) {
        return -1;
    }
    
    char c = state->data[state->pos];
    
    if (c == '"') {
        return parse_string(state);
    } else if (c == '{') {
        return parse_object(state);
    } else if (c == '[') {
        return parse_array(state);
    } else if (c == 't') {
        return parse_keyword(state, "true");
    } else if (c == 'f') {
        return parse_keyword(state, "false");
    } else if (c == 'n') {
        return parse_keyword(state, "null");
    } else if (c == '-' || (c >= '0' && c <= '9')) {
        return parse_number(state);
    }
    
    return -1;
}

int parse_json(const char *data, size_t len) {
    if (data == NULL || len == 0) {
        return -1;
    }
    
    parser_state_t state;
    state.data = data;
    state.len = len;
    state.pos = 0;
    
    if (parse_value(&state) != 0) {
        return -1;
    }
    
    skip_whitespace(&state);
    
    if (state.pos != state.len) {
        return -1;
    }
    
    return 0;
}