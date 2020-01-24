using System;

namespace SNNP.MLP
{
    public static class Activation
    {
        // Sigmoid
        public static double S(double x) => 1 / (1 + Math.Exp(-x));
        public static double DS(double x) => S(x) * (1 - S(x));

        // Relu => output: [0, infinity)
        public static double ReLU(double x) => x > 0 ? x : 0;
        public static double DReLU(double x) => x > 0 ? 1 : 0;

        // LeakyReLU => alpha = 0.01 => output: (-infinity, infinity)
        public static double LeakyReLU(double x) => (x > 0) ? x : 0.01 * x;
        public static double DLeakyReLU(double x) => (x > 0) ? 1 : 0.01;

        // SoftPlus => output: (0, infinity)
        public static double SoftPlus(double x) => Math.Log(1 + Math.Exp(x));
        public static double DSoftPlus(double x) => 1 / (1 + Math.Exp(-x));

        // BentIdentity => output: (-infinity, infinity)
        public static double BentIdentity(double x) => ((Math.Sqrt(x * x + 1) - 1) / 2) + x;
        public static double DBentIdentity(double x) => (x / (2 * Math.Sqrt(x * x + 1))) + 1;

        // Identity => output: (-infinity, infinity)
        public static double I(double x) => x;
        // Due to the MLP class' architecture, a value must be sent to the method.
        public static double DI(double _) => 1;

        // TanH => output: (-1, 1)
        public static double TanH(double x)
        {
            var ex = Math.Exp(x);
            var e_x = Math.Exp(-x);
            return (ex - e_x) / (ex + e_x);
        }

        public static double DTanH(double x)
        {
            x = TanH(x);
            return 1 - (x * x);
        }

        // ArcTan => output: (-PI/2, PI/2)
        public static double ArcTan(double x) => Math.Atan(x);
        public static double DArcTan(double x) => 1 / (x * x + 1);

        // ElliotSig / Softsign => output: (-1, 1)
        public static double ElliotSig(double x) => x / (1 + Math.Abs(x));
        public static double DElliotSig(double x)
        {
            x = 1 + Math.Abs(x);
            return x / (x * x);
        }

        // Sinusoid => output: [-1, 1]
        public static double Sinusoid(double x) => Math.Sin(x);
        public static double DSinusoid(double x) => Math.Cos(x);

        // Sinc => output: [\approx -.217234, 1]
        public static double Sinc(double x) => x != 0 ? Math.Sin(x) / x : 1;
        public static double DSinc(double x) => x != 0 ? (Math.Cos(x) / x) - (Math.Sin(x) / (x * x)) : 0;

        // Gaussian => output: (0, 1]
        public static double Gaussian(double x) => Math.Exp(-(x * x));
        public static double DGaussian(double x) => -2 * x * Math.Exp(-(x * x));
    }
}