#include "hash.h"
#include <string.h>
#include <stdint.h>

// SHA-256 constants
static const uint32_t K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

typedef struct {
    uint32_t state[8];
    uint8_t buffer[64];
    uint64_t bitlen;
    uint32_t buflen;
} SHA256_CTX;

#define ROTR(x, n) (((x) >> (n)) | ((x) << (32 - (n))))
#define CH(x, y, z) (((x) & (y)) ^ (~(x) & (z)))
#define MAJ(x, y, z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define EP0(x) (ROTR(x, 2) ^ ROTR(x, 13) ^ ROTR(x, 22))
#define EP1(x) (ROTR(x, 6) ^ ROTR(x, 11) ^ ROTR(x, 25))
#define SIG0(x) (ROTR(x, 7) ^ ROTR(x, 18) ^ ((x) >> 3))
#define SIG1(x) (ROTR(x, 17) ^ ROTR(x, 19) ^ ((x) >> 10))

static void sha256_transform(SHA256_CTX *ctx, const uint8_t data[]) {
    uint32_t a, b, c, d, e, f, g, h, i, t1, t2, m[64];

    for (i = 0; i < 16; ++i) {
        m[i] = (data[i * 4] << 24) | (data[i * 4 + 1] << 16) | (data[i * 4 + 2] << 8) | (data[i * 4 + 3]);
    }
    for (; i < 64; ++i) {
        m[i] = SIG1(m[i - 2]) + m[i - 7] + SIG0(m[i - 15]) + m[i - 16];
    }

    a = ctx->state[0];
    b = ctx->state[1];
    c = ctx->state[2];
    d = ctx->state[3];
    e = ctx->state[4];
    f = ctx->state[5];
    g = ctx->state[6];
    h = ctx->state[7];

    for (i = 0; i < 64; ++i) {
        t1 = h + EP1(e) + CH(e, f, g) + K[i] + m[i];
        t2 = EP0(a) + MAJ(a, b, c);
        h = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }

    ctx->state[0] += a;
    ctx->state[1] += b;
    ctx->state[2] += c;
    ctx->state[3] += d;
    ctx->state[4] += e;
    ctx->state[5] += f;
    ctx->state[6] += g;
    ctx->state[7] += h;
}

void sha256_init(void* context) {
    SHA256_CTX *ctx = (SHA256_CTX*)context;
    ctx->buflen = 0;
    ctx->bitlen = 0;
    ctx->state[0] = 0x6a09e667;
    ctx->state[1] = 0xbb67ae85;
    ctx->state[2] = 0x3c6ef372;
    ctx->state[3] = 0xa54ff53a;
    ctx->state[4] = 0x510e527f;
    ctx->state[5] = 0x9b05688c;
    ctx->state[6] = 0x1f83d9ab;
    ctx->state[7] = 0x5be0cd19;
}

void sha256_hash(void* context, const char* input, char* output) {
    SHA256_CTX ctx;
    sha256_init(&ctx);
    
    size_t len = strlen(input);
    uint32_t i;
    
    for (i = 0; i < len; ++i) {
        ctx.buffer[ctx.buflen] = input[i];
        ctx.buflen++;
        if (ctx.buflen == 64) {
            sha256_transform(&ctx, ctx.buffer);
            ctx.bitlen += 512;
            ctx.buflen = 0;
        }
    }
    
    i = ctx.buflen;
    if (ctx.buflen < 56) {
        ctx.buffer[i++] = 0x80;
        while (i < 56) {
            ctx.buffer[i++] = 0x00;
        }
    } else {
        ctx.buffer[i++] = 0x80;
        while (i < 64) {
            ctx.buffer[i++] = 0x00;
        }
        sha256_transform(&ctx, ctx.buffer);
        memset(ctx.buffer, 0, 56);
    }
    
    ctx.bitlen += ctx.buflen * 8;
    ctx.buffer[63] = ctx.bitlen;
    ctx.buffer[62] = ctx.bitlen >> 8;
    ctx.buffer[61] = ctx.bitlen >> 16;
    ctx.buffer[60] = ctx.bitlen >> 24;
    ctx.buffer[59] = ctx.bitlen >> 32;
    ctx.buffer[58] = ctx.bitlen >> 40;
    ctx.buffer[57] = ctx.bitlen >> 48;
    ctx.buffer[56] = ctx.bitlen >> 56;
    sha256_transform(&ctx, ctx.buffer);
    
    for (i = 0; i < 8; ++i) {
        output[i * 8 + 0] = "0123456789abcdef"[(ctx.state[i] >> 28) & 0xf];
        output[i * 8 + 1] = "0123456789abcdef"[(ctx.state[i] >> 24) & 0xf];
        output[i * 8 + 2] = "0123456789abcdef"[(ctx.state[i] >> 20) & 0xf];
        output[i * 8 + 3] = "0123456789abcdef"[(ctx.state[i] >> 16) & 0xf];
        output[i * 8 + 4] = "0123456789abcdef"[(ctx.state[i] >> 12) & 0xf];
        output[i * 8 + 5] = "0123456789abcdef"[(ctx.state[i] >> 8) & 0xf];
        output[i * 8 + 6] = "0123456789abcdef"[(ctx.state[i] >> 4) & 0xf];
        output[i * 8 + 7] = "0123456789abcdef"[(ctx.state[i]) & 0xf];
    }
    output[64] = '\0';
}