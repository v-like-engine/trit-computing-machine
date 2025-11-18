#include <gtest/gtest.h>
#include "trit.hpp"
#include "tryte.hpp"

using namespace ternary;

// Trit tests
TEST(TritTest, Construction) {
    Trit t1;
    EXPECT_EQ(t1.toInt(), 0);

    Trit t2(Trit::PLUS);
    EXPECT_EQ(t2.toInt(), 1);

    Trit t3(Trit::MINUS);
    EXPECT_EQ(t3.toInt(), -1);
}

TEST(TritTest, Arithmetic) {
    Trit plus(Trit::PLUS);
    Trit minus(Trit::MINUS);
    Trit zero(Trit::ZERO);

    // Addition
    EXPECT_EQ((plus + plus).toInt(), 2);   // Overflows to 2
    EXPECT_EQ((plus + zero).toInt(), 1);
    EXPECT_EQ((plus + minus).toInt(), 0);

    // Subtraction
    EXPECT_EQ((plus - minus).toInt(), 2);
    EXPECT_EQ((zero - plus).toInt(), -1);

    // Multiplication
    EXPECT_EQ((plus * plus).toInt(), 1);
    EXPECT_EQ((plus * minus).toInt(), -1);
    EXPECT_EQ((minus * minus).toInt(), 1);

    // Negation
    EXPECT_EQ((-plus).toInt(), -1);
    EXPECT_EQ((-minus).toInt(), 1);
    EXPECT_EQ((-zero).toInt(), 0);
}

TEST(TritTest, Logic) {
    Trit plus(Trit::PLUS);
    Trit minus(Trit::MINUS);
    Trit zero(Trit::ZERO);

    // AND (min)
    EXPECT_EQ(plus.logicAnd(minus).toInt(), -1);
    EXPECT_EQ(plus.logicAnd(zero).toInt(), 0);
    EXPECT_EQ(zero.logicAnd(minus).toInt(), -1);

    // OR (max)
    EXPECT_EQ(plus.logicOr(minus).toInt(), 1);
    EXPECT_EQ(plus.logicOr(zero).toInt(), 1);
    EXPECT_EQ(zero.logicOr(minus).toInt(), 0);

    // NOT
    EXPECT_EQ(plus.logicNot().toInt(), -1);
    EXPECT_EQ(minus.logicNot().toInt(), 1);
    EXPECT_EQ(zero.logicNot().toInt(), 0);
}

// Tryte tests
TEST(TryteTest, Construction) {
    Tryte t1;
    EXPECT_EQ(t1.toInt64(), 0);

    Tryte t2(42);
    EXPECT_EQ(t2.toInt64(), 42);

    Tryte t3(-17);
    EXPECT_EQ(t3.toInt64(), -17);
}

TEST(TryteTest, BalancedTernary) {
    // Test conversion to/from balanced ternary
    Tryte t1(5);  // 5 = 1*3^1 + 2*3^0 = 3 + 2, but in balanced: +-+
    // Actually: 5 = 1*9 + (-1)*3 + (-1)*1 = 9 - 3 - 1 = 5
    // Or: 5 = 2*3 + (-1)*1 = 6 - 1, needs carry...
    // 5 in balanced ternary is: +--+ (1*9 + (-1)*3 + (-1)*1 = 9-3-1=5) NO
    // Let me recalculate: 5 = +--+ means 1 + (-3) + (-9) NO
    // 5 = 1*1 + (-1)*3 + 1*9 = 1 - 3 + 9 = 7 NO
    // Actually in our order (LST first): positions are 1, 3, 9, 27...
    // 5 in balanced: 1*9 + (-1)*3 + (-1)*1 = 9-3-1 = 5
    // So: trit[0]=-, trit[1]=-, trit[2]=+, rest=0
    // String from high to low: 000000000000000+--

    EXPECT_EQ(t1.toInt64(), 5);

    // Test zero
    Tryte t2(0);
    EXPECT_EQ(t2.toBalancedTernaryString(), "000000000000000000");
}

TEST(TryteTest, Arithmetic) {
    Tryte t1(10);
    Tryte t2(5);

    Tryte sum = t1 + t2;
    EXPECT_EQ(sum.toInt64(), 15);

    Tryte diff = t1 - t2;
    EXPECT_EQ(diff.toInt64(), 5);

    Tryte prod = t1 * t2;
    EXPECT_EQ(prod.toInt64(), 50);

    Tryte neg = -t1;
    EXPECT_EQ(neg.toInt64(), -10);
}

TEST(TryteTest, Negation) {
    // One of the key advantages: easy negation
    Tryte t(42);
    Tryte neg = -t;

    EXPECT_EQ(t.toInt64(), 42);
    EXPECT_EQ(neg.toInt64(), -42);

    // Double negation
    Tryte double_neg = -neg;
    EXPECT_EQ(double_neg.toInt64(), 42);
}

TEST(TryteTest, Range) {
    int64_t max_val = Tryte::maxValue();
    int64_t min_val = Tryte::minValue();

    EXPECT_GT(max_val, 0);
    EXPECT_LT(min_val, 0);
    EXPECT_EQ(max_val, -min_val);

    // Should be (3^18 - 1) / 2 = 193710244
    EXPECT_EQ(max_val, 193710244);
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
