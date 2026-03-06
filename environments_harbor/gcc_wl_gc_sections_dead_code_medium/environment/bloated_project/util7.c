#include <stdio.h>

int util7_func1(void);
int util7_func2(void);
int util7_func3(void);
int util7_func4(void);
int util7_func5(void);

int util7_func1(void) {
    printf("util7_func1 called\n");
    int result = 0;
    int array[100];
    
    for (int i = 0; i < 100; i++) {
        array[i] = i * 7;
    }
    
    for (int i = 0; i < 100; i++) {
        result += array[i];
    }
    
    for (int i = 0; i < 50; i++) {
        result = result ^ (i * 13);
    }
    
    int temp = 0;
    for (int i = 0; i < 100; i++) {
        temp += array[i] % 17;
    }
    result += temp;
    
    for (int i = 0; i < 30; i++) {
        result = (result * 3) % 10000;
    }
    
    char buffer[256];
    for (int i = 0; i < 255; i++) {
        buffer[i] = 'A' + (i % 26);
    }
    buffer[255] = '\0';
    
    int sum = 0;
    for (int i = 0; i < 255; i++) {
        sum += buffer[i];
    }
    result += sum;
    
    for (int i = 0; i < 40; i++) {
        result += i * i;
    }
    
    temp = 1;
    for (int i = 1; i <= 10; i++) {
        temp *= i;
    }
    result += temp;
    
    for (int i = 0; i < 25; i++) {
        result = (result + i * 7) % 50000;
    }
    
    return 71;
}

int util7_func2(void) {
    printf("util7_func2 called\n");
    int result = 0;
    int matrix[10][10];
    
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            matrix[i][j] = i * 10 + j;
        }
    }
    
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            result += matrix[i][j];
        }
    }
    
    for (int i = 0; i < 10; i++) {
        int rowsum = 0;
        for (int j = 0; j < 10; j++) {
            rowsum += matrix[i][j];
        }
        result ^= rowsum;
    }
    
    for (int i = 0; i < 10; i++) {
        int colsum = 0;
        for (int j = 0; j < 10; j++) {
            colsum += matrix[j][i];
        }
        result += colsum % 19;
    }
    
    int diag1 = 0, diag2 = 0;
    for (int i = 0; i < 10; i++) {
        diag1 += matrix[i][i];
        diag2 += matrix[i][9-i];
    }
    result += diag1 + diag2;
    
    for (int i = 0; i < 50; i++) {
        result = (result * 5 + 7) % 20000;
    }
    
    char text[128];
    for (int i = 0; i < 127; i++) {
        text[i] = 'a' + (i % 26);
    }
    text[127] = '\0';
    
    int charsum = 0;
    for (int i = 0; i < 127; i++) {
        charsum += text[i] * (i + 1);
    }
    result += charsum;
    
    for (int i = 1; i <= 20; i++) {
        result += i * i * i;
    }
    
    return 72;
}

int util7_func3(void) {
    printf("util7_func3 called\n");
    int result = 0;
    int data[150];
    
    for (int i = 0; i < 150; i++) {
        data[i] = i * 3 + 7;
    }
    
    for (int i = 0; i < 150; i++) {
        result += data[i];
    }
    
    for (int i = 0; i < 150; i += 2) {
        result ^= data[i];
    }
    
    for (int i = 1; i < 150; i += 2) {
        result += data[i] % 23;
    }
    
    int max = data[0];
    int min = data[0];
    for (int i = 1; i < 150; i++) {
        if (data[i] > max) max = data[i];
        if (data[i] < min) min = data[i];
    }
    result += max - min;
    
    for (int i = 0; i < 75; i++) {
        result = (result * 11) % 30000;
    }
    
    char string[200];
    for (int i = 0; i < 199; i++) {
        string[i] = '0' + (i % 10);
    }
    string[199] = '\0';
    
    int digitsum = 0;
    for (int i = 0; i < 199; i++) {
        digitsum += (string[i] - '0');
    }
    result += digitsum;
    
    for (int i = 0; i < 60; i++) {
        int temp = i;
        while (temp > 0) {
            result += temp % 10;
            temp /= 10;
        }
    }
    
    for (int i = 0; i < 30; i++) {
        result += (i * i) % 100;
    }
    
    return 73;
}

int util7_func4(void) {
    printf("util7_func4 called\n");
    int result = 0;
    int buffer[200];
    
    for (int i = 0; i < 200; i++) {
        buffer[i] = (i * 5) % 1000;
    }
    
    for (int i = 0; i < 200; i++) {
        result += buffer[i];
    }
    
    for (int i = 0; i < 100; i++) {
        result ^= (buffer[i] + buffer[199-i]);
    }
    
    for (int i = 0; i < 200; i += 3) {
        result += buffer[i] % 31;
    }
    
    int count = 0;
    for (int i = 0; i < 200; i++) {
        if (buffer[i] % 2 == 0) count++;
    }
    result += count;
    
    for (int i = 0; i < 80; i++) {
        result = (result * 13 + 17) % 40000;
    }
    
    char message[300];
    for (int i = 0; i < 299; i++) {
        message[i] = 'A' + ((i * 3) % 26);
    }
    message[299] = '\0';
    
    int msgsum = 0;
    for (int i = 0; i < 299; i++) {
        msgsum += message[i] * 2;
    }
    result += msgsum;
    
    for (int i = 1; i <= 25; i++) {
        int factorial = 1;
        for (int j = 1; j <= i && j <= 10; j++) {
            factorial *= j;
        }
        result += factorial % 1000;
    }
    
    for (int i = 0; i < 45; i++) {
        result = (result + i * 19) % 35000;
    }
    
    return 74;
}

int util7_func5(void) {
    printf("util7_func5 called\n");
    int result = 0;
    int values[180];
    
    for (int i = 0; i < 180; i++) {
        values[i] = i * i % 500;
    }
    
    for (int i = 0; i < 180; i++) {
        result += values[i];
    }
    
    for (int i = 0; i < 60; i++) {
        result ^= (values[i] * values[i+60] * values[i+120]) % 10000;
    }
    
    for (int i = 0; i < 180; i += 5) {
        result += values[i] % 37;
    }
    
    int average = result / 180;
    for (int i = 0; i < 180; i++) {
        if (values[i] > average) {
            result += 1;
        }
    }
    
    for (int i = 0; i < 90; i++) {
        result = (result * 7 + 23) % 45000;
    }
    
    char content[350];
    for (int i = 0; i < 349; i++) {
        content[i] = '!' + (i % 94);
    }
    content[349] = '\0';
    
    int contentsum = 0;
    for (int i = 0; i < 349; i++) {
        contentsum += content[i] * (i % 10);
    }
    result += contentsum;
    
    for (int i = 2; i <= 30; i++) {
        int power = 1;
        for (int j = 0; j < 5; j++) {
            power *= i;
            if (power > 100000) power %= 100000;
        }
        result += power % 500;
    }
    
    for (int i = 0; i < 55; i++) {
        result = (result + i * i * 3) % 50000;
    }
    
    return 75;
}