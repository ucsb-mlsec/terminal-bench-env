#include "config.h"
#include <string.h>
#include <stddef.h>

// Used configuration arrays
const int config_values[32] = {
    100, 200, 300, 400, 500, 600, 700, 800,
    900, 1000, 1100, 1200, 1300, 1400, 1500, 1600,
    1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400,
    2500, 2600, 2700, 2800, 2900, 3000, 3100, 3200
};

const char* lookup_table[16] = {
    "alpha", "beta", "gamma", "delta",
    "epsilon", "zeta", "eta", "theta",
    "iota", "kappa", "lambda", "mu",
    "nu", "xi", "omicron", "pi"
};

const double scaling_factors[24] = {
    1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7,
    1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5,
    2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3
};

// Unused legacy configuration arrays
const int old_config_data[80] = {
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
    17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
    33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
    49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64,
    65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80
};

const char* deprecated_lookup[100] = {
    "deprecated_001", "deprecated_002", "deprecated_003", "deprecated_004", "deprecated_005",
    "deprecated_006", "deprecated_007", "deprecated_008", "deprecated_009", "deprecated_010",
    "deprecated_011", "deprecated_012", "deprecated_013", "deprecated_014", "deprecated_015",
    "deprecated_016", "deprecated_017", "deprecated_018", "deprecated_019", "deprecated_020",
    "deprecated_021", "deprecated_022", "deprecated_023", "deprecated_024", "deprecated_025",
    "deprecated_026", "deprecated_027", "deprecated_028", "deprecated_029", "deprecated_030",
    "deprecated_031", "deprecated_032", "deprecated_033", "deprecated_034", "deprecated_035",
    "deprecated_036", "deprecated_037", "deprecated_038", "deprecated_039", "deprecated_040",
    "deprecated_041", "deprecated_042", "deprecated_043", "deprecated_044", "deprecated_045",
    "deprecated_046", "deprecated_047", "deprecated_048", "deprecated_049", "deprecated_050",
    "deprecated_051", "deprecated_052", "deprecated_053", "deprecated_054", "deprecated_055",
    "deprecated_056", "deprecated_057", "deprecated_058", "deprecated_059", "deprecated_060",
    "deprecated_061", "deprecated_062", "deprecated_063", "deprecated_064", "deprecated_065",
    "deprecated_066", "deprecated_067", "deprecated_068", "deprecated_069", "deprecated_070",
    "deprecated_071", "deprecated_072", "deprecated_073", "deprecated_074", "deprecated_075",
    "deprecated_076", "deprecated_077", "deprecated_078", "deprecated_079", "deprecated_080",
    "deprecated_081", "deprecated_082", "deprecated_083", "deprecated_084", "deprecated_085",
    "deprecated_086", "deprecated_087", "deprecated_088", "deprecated_089", "deprecated_090",
    "deprecated_091", "deprecated_092", "deprecated_093", "deprecated_094", "deprecated_095",
    "deprecated_096", "deprecated_097", "deprecated_098", "deprecated_099", "deprecated_100"
};

const double unused_mapping[75] = {
    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
    1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0,
    2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0,
    3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0,
    4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0,
    5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0,
    6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 7.0,
    7.1, 7.2, 7.3, 7.4, 7.5
};

const long legacy_offsets[60] = {
    1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000,
    11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000, 19000, 20000,
    21000, 22000, 23000, 24000, 25000, 26000, 27000, 28000, 29000, 30000,
    31000, 32000, 33000, 34000, 35000, 36000, 37000, 38000, 39000, 40000,
    41000, 42000, 43000, 44000, 45000, 46000, 47000, 48000, 49000, 50000,
    51000, 52000, 53000, 54000, 55000, 56000, 57000, 58000, 59000, 60000
};

const int obsolete_parameters[90] = {
    10, 20, 30, 40, 50, 60, 70, 80, 90, 100,
    110, 120, 130, 140, 150, 160, 170, 180, 190, 200,
    210, 220, 230, 240, 250, 260, 270, 280, 290, 300,
    310, 320, 330, 340, 350, 360, 370, 380, 390, 400,
    410, 420, 430, 440, 450, 460, 470, 480, 490, 500,
    510, 520, 530, 540, 550, 560, 570, 580, 590, 600,
    610, 620, 630, 640, 650, 660, 670, 680, 690, 700,
    710, 720, 730, 740, 750, 760, 770, 780, 790, 800,
    810, 820, 830, 840, 850, 860, 870, 880, 890, 900
};

const char* unused_string_table[70] = {
    "unused_00", "unused_01", "unused_02", "unused_03", "unused_04",
    "unused_05", "unused_06", "unused_07", "unused_08", "unused_09",
    "unused_10", "unused_11", "unused_12", "unused_13", "unused_14",
    "unused_15", "unused_16", "unused_17", "unused_18", "unused_19",
    "unused_20", "unused_21", "unused_22", "unused_23", "unused_24",
    "unused_25", "unused_26", "unused_27", "unused_28", "unused_29",
    "unused_30", "unused_31", "unused_32", "unused_33", "unused_34",
    "unused_35", "unused_36", "unused_37", "unused_38", "unused_39",
    "unused_40", "unused_41", "unused_42", "unused_43", "unused_44",
    "unused_45", "unused_46", "unused_47", "unused_48", "unused_49",
    "unused_50", "unused_51", "unused_52", "unused_53", "unused_54",
    "unused_55", "unused_56", "unused_57", "unused_58", "unused_59",
    "unused_60", "unused_61", "unused_62", "unused_63", "unused_64",
    "unused_65", "unused_66", "unused_67", "unused_68", "unused_69"
};

const float retired_coefficients[85] = {
    0.01f, 0.02f, 0.03f, 0.04f, 0.05f, 0.06f, 0.07f, 0.08f, 0.09f, 0.10f,
    0.11f, 0.12f, 0.13f, 0.14f, 0.15f, 0.16f, 0.17f, 0.18f, 0.19f, 0.20f,
    0.21f, 0.22f, 0.23f, 0.24f, 0.25f, 0.26f, 0.27f, 0.28f, 0.29f, 0.30f,
    0.31f, 0.32f, 0.33f, 0.34f, 0.35f, 0.36f, 0.37f, 0.38f, 0.39f, 0.40f,
    0.41f, 0.42f, 0.43f, 0.44f, 0.45f, 0.46f, 0.47f, 0.48f, 0.49f, 0.50f,
    0.51f, 0.52f, 0.53f, 0.54f, 0.55f, 0.56f, 0.57f, 0.58f, 0.59f, 0.60f,
    0.61f, 0.62f, 0.63f, 0.64f, 0.65f, 0.66f, 0.67f, 0.68f, 0.69f, 0.70f,
    0.71f, 0.72f, 0.73f, 0.74f, 0.75f, 0.76f, 0.77f, 0.78f, 0.79f, 0.80f,
    0.81f, 0.82f, 0.83f, 0.84f, 0.85f
};

// Used functions
int get_config_value(int index) {
    if (index < 0 || index >= 32) {
        return -1;
    }
    return config_values[index];
}

const char* lookup_string(int index) {
    if (index < 0 || index >= 16) {
        return NULL;
    }
    return lookup_table[index];
}

double get_scaling_factor(int index) {
    if (index < 0 || index >= 24) {
        return 1.0;
    }
    return scaling_factors[index];
}

// Unused functions
int get_old_config(int index) {
    if (index < 0 || index >= 80) {
        return -1;
    }
    return old_config_data[index];
}

const char* deprecated_lookup_func(int index) {
    if (index < 0 || index >= 100) {
        return NULL;
    }
    return deprecated_lookup[index];
}

double get_unused_mapping(int index) {
    if (index < 0 || index >= 75) {
        return 0.0;
    }
    return unused_mapping[index];
}

long get_legacy_offset(int index) {
    if (index < 0 || index >= 60) {
        return -1;
    }
    return legacy_offsets[index];
}

int get_obsolete_parameter(int index) {
    if (index < 0 || index >= 90) {
        return -1;
    }
    return obsolete_parameters[index];
}