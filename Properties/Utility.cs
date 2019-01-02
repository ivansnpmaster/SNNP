using System;

public static class Utility
{
    public static double NextDouble(double minValue, double maxValue) => new Random().NextDouble() * (maxValue - minValue) + minValue;
}