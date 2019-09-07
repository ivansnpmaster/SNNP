using System;

namespace SNNP.MLP
{
    public static class Activation
    {
        public static double S(double x) => 1 / (1 + Math.Exp(-x));

        // Assuming y as fnet
        public static double DS(double y) => y * (1 - y);
    }
}