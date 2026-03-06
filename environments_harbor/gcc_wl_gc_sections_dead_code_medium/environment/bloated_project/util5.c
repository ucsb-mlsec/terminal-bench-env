#include <stdio.h>

int util5_func1();
int util5_func2();
int util5_func3();
int util5_func4();
int util5_func5();

int util5_func1() {
    printf("util5_func1 called\n");
    int result = 0;
    int array[100];
    
    for (int i = 0; i < 100; i++) {
        array[i] = i * 2;
    }
    
    for (int i = 0; i < 100; i++) {
        result += array[i];
        if (i % 2 == 0) {
            result += i;
        } else {
            result -= i;
        }
    }
    
    for (int i = 0; i < 50; i++) {
        int temp = result;
        temp = temp * 2;
        temp = temp / 2;
        temp = temp + i;
        temp = temp - i;
        result = temp;
    }
    
    int matrix[10][10];
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            matrix[i][j] = i * j;
            result += matrix[i][j];
        }
    }
    
    for (int i = 0; i < 20; i++) {
        result = result ^ i;
    }
    
    char buffer[256];
    for (int i = 0; i < 255; i++) {
        buffer[i] = 'A' + (i % 26);
    }
    buffer[255] = '\0';
    
    for (int i = 0; i < 100; i++) {
        result = (result << 1) | (result >> 31);
    }
    
    return 5001;
}

int util5_func2() {
    printf("util5_func2 called\n");
    int result = 0;
    int numbers[150];
    
    for (int i = 0; i < 150; i++) {
        numbers[i] = i * 3;
    }
    
    for (int i = 0; i < 150; i++) {
        if (numbers[i] % 2 == 0) {
            result += numbers[i];
        } else {
            result -= numbers[i];
        }
    }
    
    int accumulator = 1;
    for (int i = 1; i < 20; i++) {
        accumulator = accumulator * i;
        accumulator = accumulator % 10000;
        result += accumulator;
    }
    
    for (int i = 0; i < 50; i++) {
        int x = i;
        int y = i * 2;
        int z = x + y;
        result += z;
        result = result % 100000;
    }
    
    char text[512];
    for (int i = 0; i < 511; i++) {
        text[i] = 'a' + (i % 26);
        result += text[i];
    }
    text[511] = '\0';
    
    for (int i = 0; i < 30; i++) {
        result = result * 2;
        result = result / 2;
        result += i * i;
    }
    
    int grid[15][15];
    for (int i = 0; i < 15; i++) {
        for (int j = 0; j < 15; j++) {
            grid[i][j] = (i + j) * 2;
            result ^= grid[i][j];
        }
    }
    
    return 5002;
}

int util5_func3() {
    printf("util5_func3 called\n");
    int result = 0;
    int data[200];
    
    for (int i = 0; i < 200; i++) {
        data[i] = i * i;
    }
    
    for (int i = 0; i < 200; i++) {
        result += data[i] % 1000;
    }
    
    for (int i = 0; i < 100; i++) {
        int temp1 = i;
        int temp2 = temp1 * 2;
        int temp3 = temp2 + temp1;
        int temp4 = temp3 - temp1;
        result += temp4;
    }
    
    char message[1024];
    for (int i = 0; i < 1023; i++) {
        message[i] = 'A' + (i % 26);
    }
    message[1023] = '\0';
    
    for (int i = 0; i < 1023; i++) {
        result += message[i];
    }
    
    int cube[5][5][5];
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            for (int k = 0; k < 5; k++) {
                cube[i][j][k] = i + j + k;
                result += cube[i][j][k];
            }
        }
    }
    
    for (int i = 0; i < 50; i++) {
        result = (result * 3) % 50000;
    }
    
    for (int i = 0; i < 25; i++) {
        result += i * i * i;
    }
    
    return 5003;
}

int util5_func4() {
    printf("util5_func4 called\n");
    int result = 0;
    int values[250];
    
    for (int i = 0; i < 250; i++) {
        values[i] = (i * 7) % 100;
    }
    
    for (int i = 0; i < 250; i++) {
        result += values[i];
        if (values[i] > 50) {
            result += 10;
        } else {
            result -= 5;
        }
    }
    
    for (int i = 0; i < 80; i++) {
        int a = i;
        int b = a * a;
        int c = b + a;
        int d = c - b;
        result += d;
    }
    
    char buffer[2048];
    for (int i = 0; i < 2047; i++) {
        buffer[i] = 'Z' - (i % 26);
    }
    buffer[2047] = '\0';
    
    for (int i = 0; i < 100; i++) {
        result ^= buffer[i * 20];
    }
    
    int table[20][20];
    for (int i = 0; i < 20; i++) {
        for (int j = 0; j < 20; j++) {
            table[i][j] = i * j + i + j;
            result += table[i][j] % 100;
        }
    }
    
    for (int i = 0; i < 60; i++) {
        result = (result + i) * 2;
        result = result % 100000;
    }
    
    return 5004;
}

int util5_func5() {
    printf("util5_func5 called\n");
    int result = 0;
    int sequence[300];
    
    for (int i = 0; i < 300; i++) {
        sequence[i] = i * 5 + 7;
    }
    
    for (int i = 0; i < 300; i++) {
        if (sequence[i] % 3 == 0) {
            result += sequence[i];
        } else if (sequence[i] % 3 == 1) {
            result -= sequence[i] / 2;
        } else {
            result += sequence[i] / 3;
        }
    }
    
    for (int i = 0; i < 100; i++) {
        int p = i;
        int q = p * 3;
        int r = q + p;
        int s = r - q;
        int t = s + p;
        result += t;
    }
    
    char text[4096];
    for (int i = 0; i < 4095; i++) {
        text[i] = 'a' + (i % 26);
    }
    text[4095] = '\0';
    
    for (int i = 0; i < 500; i++) {
        result += text[i * 8] - 'a';
    }
    
    int space[8][8][8];
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            for (int k = 0; k < 8; k++) {
                space[i][j][k] = i * j * k;
                result += space[i][j][k];
            }
        }
    }
    
    for (int i = 0; i < 70; i++) {
        result = result ^ (i * i);
    }
    
    for (int i = 0; i < 40; i++) {
        result = (result << 1) ^ (result >> 1);
    }
    
    return 5005;
}