void test_entropy_f32();
void test_logsumexp_f32();
void test_kullback_leibler_f32();
void test_logsumexp_dot_prod_f32();

// Pattern IDs
static const int INPUT1_F32_ID=0;
static const int DIM1_S16_ID=1;
static const int REF1_ENTROPY_F32_ID=2;
static const int INPUT2_F32_ID=3;
static const int DIM2_S16_ID=4;
static const int REF2_LOGSUMEXP_F32_ID=5;
static const int INPUTA3_F32_ID=6;
static const int INPUTB3_F32_ID=7;
static const int DIM3_S16_ID=8;
static const int REF3_KL_F32_ID=9;
static const int INPUTA4_F32_ID=10;
static const int INPUTB4_F32_ID=11;
static const int DIM4_S16_ID=12;
static const int REF4_LOGSUMEXP_DOT_F32_ID=13;

// Output IDs
static const int OUT_F32_ID=0;
static const int TMP_F32_ID=1;

// Test IDs
static const int TEST_ENTROPY_F32_1=1;
static const int TEST_LOGSUMEXP_F32_2=2;
static const int TEST_KULLBACK_LEIBLER_F32_3=3;
static const int TEST_LOGSUMEXP_DOT_PROD_F32_4=4;
