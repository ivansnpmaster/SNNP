using System;

namespace SNNP.MLP
{
    public static class Activation
    {
        public static double S(double x) => 1 / (1 + Math.Exp(-x));

        public static double DS(double x) => S(x) * (1 - S(x));

        // alpha = 0.1
        public static double LeakyReLU(double x) => (x > 0) ? x : 0.1 * x;
        // alpha = 0.1
        public static double DLeakyReLU(double x) => (x > 0) ? 1 : 0.1;
    }
}