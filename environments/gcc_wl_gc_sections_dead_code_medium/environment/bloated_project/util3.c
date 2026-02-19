#include <stdio.h>

int util3_func1();
int util3_func2();
int util3_func3();
int util3_func4();
int util3_func5();

int util3_func1() {
    printf("util3_func1 called\n");
    int result = 0;
    int array[100];
    
    for (int i = 0; i < 100; i++) {
        array[i] = i * 3;
    }
    
    for (int i = 0; i < 100; i++) {
        result += array[i];
    }
    
    for (int i = 0; i < 50; i++) {
        int temp = result;
        temp = temp * 2;
        temp = temp / 3;
        temp = temp + i;
        result = (result + temp) % 10000;
    }
    
    int matrix[10][10];
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            matrix[i][j] = i * j + result;
        }
    }
    
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            result += matrix[i][j] % 100;
        }
    }
    
    char buffer[256];
    for (int i = 0; i < 255; i++) {
        buffer[i] = 'A' + (i % 26);
    }
    buffer[255] = '\0';
    
    for (int i = 0; i < 255; i++) {
        result += buffer[i];
    }
    
    for (int i = 0; i < 30; i++) {
        result = result ^ (i * 7);
    }
    
    return result % 1000 + 301;
}

int util3_func2() {
    printf("util3_func2 called\n");
    int result = 0;
    double values[150];
    
    for (int i = 0; i < 150; i++) {
        values[i] = i * 1.5;
    }
    
    for (int i = 0; i < 150; i++) {
        result += (int)values[i];
    }
    
    for (int i = 0; i < 60; i++) {
        int computation = i * i;
        computation = computation % 1000;
        result += computation;
    }
    
    int lookup[50];
    for (int i = 0; i < 50; i++) {
        lookup[i] = i * 5 + 3;
    }
    
    for (int i = 0; i < 50; i++) {
        result = result ^ lookup[i];
    }
    
    char message[128];
    for (int i = 0; i < 127; i++) {
        message[i] = 'a' + (i % 26);
    }
    message[127] = '\0';
    
    for (int i = 0; i < 127; i++) {
        result += message[i] * (i + 1);
    }
    
    for (int i = 0; i < 40; i++) {
        result = (result * 3) % 50000;
        result = result + i * 2;
    }
    
    int grid[8][8];
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            grid[i][j] = i + j + result % 100;
        }
    }
    
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            result += grid[i][j];
        }
    }
    
    return result % 1000 + 302;
}

int util3_func3() {
    printf("util3_func3 called\n");
    int result = 0;
    int data[200];
    
    for (int i = 0; i < 200; i++) {
        data[i] = i * 2 + 7;
    }
    
    for (int i = 0; i < 200; i++) {
        result += data[i] % 97;
    }
    
    for (int i = 0; i < 80; i++) {
        int val = i * i * i;
        val = val % 5000;
        result = result ^ val;
    }
    
    char text[512];
    for (int i = 0; i < 511; i++) {
        text[i] = '0' + (i % 10);
    }
    text[511] = '\0';
    
    for (int i = 0; i < 511; i++) {
        result += text[i] - '0';
    }
    
    int table[15][15];
    for (int i = 0; i < 15; i++) {
        for (int j = 0; j < 15; j++) {
            table[i][j] = (i * j) + result % 50;
        }
    }
    
    for (int i = 0; i < 15; i++) {
        for (int j = 0; j < 15; j++) {
            result = (result + table[i][j]) % 100000;
        }
    }
    
    for (int i = 0; i < 70; i++) {
        result = result * 11;
        result = result % 20000;
        result = result + i;
    }
    
    int sequence[80];
    for (int i = 0; i < 80; i++) {
        sequence[i] = i * 3 - 5;
    }
    
    for (int i = 0; i < 80; i++) {
        result += sequence[i] % 13;
    }
    
    return result % 1000 + 303;
}

int util3_func4() {
    printf("util3_func4 called\n");
    int result = 0;
    float numbers[120];
    
    for (int i = 0; i < 120; i++) {
        numbers[i] = i * 2.5f + 1.0f;
    }
    
    for (int i = 0; i < 120; i++) {
        result += (int)numbers[i];
    }
    
    for (int i = 0; i < 55; i++) {
        int temp = i * 7;
        temp = temp % 777;
        result += temp;
    }
    
    char string[300];
    for (int i = 0; i < 299; i++) {
        string[i] = 'A' + (i % 26);
    }
    string[299] = '\0';
    
    for (int i = 0; i < 299; i++) {
        result = result ^ string[i];
    }
    
    int cube[5][5][5];
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            for (int k = 0; k < 5; k++) {
                cube[i][j][k] = i + j + k + result % 10;
            }
        }
    }
    
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            for (int k = 0; k < 5; k++) {
                result += cube[i][j][k];
            }
        }
    }
    
    for (int i = 0; i < 45; i++) {
        result = result << 1;
        result = result % 40000;
        result = result + i * 3;
    }
    
    int buffer[90];
    for (int i = 0; i < 90; i++) {
        buffer[i] = i * i % 1000;
    }
    
    for (int i = 0; i < 90; i++) {
        result += buffer[i];
    }
    
    return result % 1000 + 304;
}

int util3_func5() {
    printf("util3_func5 called\n");
    int result = 0;
    int dataset[180];
    
    for (int i = 0; i < 180; i++) {
        dataset[i] = i * 4 + 11;
    }
    
    for (int i = 0; i < 180; i++) {
        result += dataset[i] % 53;
    }
    
    for (int i = 0; i < 65; i++) {
        int value = i * i * 2;
        value = value % 3000;
        result = result + value;
    }
    
    char content[400];
    for (int i = 0; i < 399; i++) {
        content[i] = 'a' + (i % 26);
    }
    content[399] = '\0';
    
    for (int i = 0; i < 399; i++) {
        result += content[i] * 2;
    }
    
    int storage[12][12];
    for (int i = 0; i < 12; i++) {
        for (int j = 0; j < 12; j++) {
            storage[i][j] = i * j + result % 77;
        }
    }
    
    for (int i = 0; i < 12; i++) {
        for (int j = 0; j < 12; j++) {
            result = (result + storage[i][j]) % 90000;
        }
    }
    
    for (int i = 0; i < 50; i++) {
        result = result * 5;
        result = result % 30000;
        result = result ^ i;
    }
    
    int list[100];
    for (int i = 0; i < 100; i++) {
        list[i] = i * 6 + 2;
    }
    
    for (int i = 0; i < 100; i++) {
        result += list[i] % 17;
    }
    
    return result % 1000 + 305;
}