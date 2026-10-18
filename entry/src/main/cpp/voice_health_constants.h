// AUTO-GENERATED from voice_health_model_gbdt.json + the 57 expert heads.
#pragma once
static const int   VH_N_CODES    = 57;
static const int   VH_N_RAW      = 135;
static const int   VH_N_FEATURES = 55;
static const float VH_THRESHOLD  = 0.30177321f;
// 'c' = CE head (7 logits, argmax) | 'h' = Huber head (1 value, round+clip 0..6)
static const char VH_KINDS[57] = {'h','h','c','h','h','c','h','c','h','h','h','h','h','h','c','c','c','h','h','h','h','h','c','h','c','h','c','h','h','h','h','h','h','h','h','h','h','h','h','h','h','h','c','h','h','h','h','h','c','h','h','h','h','c','h','c','h'};
static const int VH_OFFSETS[57] = {0,1,2,9,10,11,18,19,26,27,28,29,30,31,32,39,46,53,54,55,56,57,58,65,66,73,74,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,103,104,105,106,107,108,115,116,117,118,119,126,127,134};
// feature order of the GBDT: -1 = speech_probability, else index into the 57 codes
static const int VH_FEATURE_CODE[55] = {-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,17,18,27,29,30,19,20,21,22,23,24,25,46,47,48,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,49,50,51,52,53,54,55,56};
static const char* VH_CODES[57] = {"AGEV","AROU","ARSH","ATCK","BKGN","BRGT","CHNK","CLRT","COGL","DARC","DFLU","EMPH","ESTH","EXPL","FOCS","FULL","GEND","HARM","METL","R_CHST","R_HEAD","R_MASK","R_MIXD","R_NASL","R_ORAL","R_THRT","RANG","RCQL","REGS","RESP","ROUG","S_ASMR","S_AUTH","S_CART","S_CASU","S_CONV","S_DRAM","S_FORM","S_MONO","S_NARR","S_NEWS","S_PLAY","S_RANT","S_STRY","S_TECH","S_WHIS","SMTH","STNC","STRU","TEMP","TENS","VALN","VALS","VFLX","VOLT","VULN","WARM"};
