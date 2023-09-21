#include <stdio.h>
#include<stdlib.h>

// Binary threshold function
int b(int a, int b, int c) {
    int positive_count = 0;

    if (a > 0) {
        positive_count++;
    }
    if (b > 0) {
        positive_count++;
    }
    if (c > 0) {
        positive_count++;
    }

    return (positive_count >= 2) ? 1 : 0;
}
int inttobin(){
    const char* binaryString = "01101101";
    char* endptr; // Pointer to the first character not converted

    // Convert binary string to integer
    long int intValue = strtol(binaryString, &endptr, 2);

    // Check for conversion errors
    if (*endptr != '\0') {
        printf("Conversion failed. Invalid binary string.\n");
        return 1;
    }

    printf("Integer value: %ld\n", intValue);
}


int main() {
    int P5=101,P6= 120,P7= 130,P4=90,C= 100,P0= 120,P3=114,P2= 80,P1= 70;
    int matrix[3][3] = {
        {P7, P0, P1},
        {P6, C, P2},
        {P5, P4, P3}
    };
    
    int CP[4];
    
    CP[0] = b(P7-C,P0-C,P1-C);
    CP[1] = b(P1-C,P2-C,P3-C);
    CP[2] = b(P3-C,P4-C,P5-C);
    CP[3] = b(P5-C,P6-C,P7-C);
    CP[4] = b((P6 - P0), (C - P0), (P2 - P0));
    CP[5] = b((P4 - P2), (C - P2), (P0 - P2));
    CP[6] = b((P2 - P4), (C - P4), (P6 - P4));
    CP[7] = b((P0 - P6), (C - P6), (P4 - P6));
    

    
    printf("CP0 = %d\n", CP[0]);
    printf("CP1 = %d\n", CP[1]);
    printf("CP2 = %d\n", CP[2]);
    printf("CP3 = %d\n", CP[3]);
    printf("CP4 = %d\n", CP[4]);
    printf("CP5 = %d\n", CP[5]);
    printf("CP6 = %d\n", CP[6]);
    printf("CP7 = %d\n", CP[7]);
    inttobin();
    
    return 0;
}
