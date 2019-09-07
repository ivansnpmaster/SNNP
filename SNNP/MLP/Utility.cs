using System;

namespace SNNP.MLP
{
    public static class Utility
    {
        private static readonly Random random = new Random();
        private static readonly object syncLock = new object();

        public static double NextDouble(double minValue, double maxValue)
        {
            return random.NextDouble() * (maxValue - minValue) + minValue;
        }

        public static int Next(int minValue, int maxValue)
        {
            return random.Next(minValue, maxValue);
        }
    }
}