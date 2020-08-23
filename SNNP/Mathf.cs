using System;
using SNNP.MLP;

namespace SNNP
{
    public static class Mathf
    {
        private static readonly Random random = new Random();

        public static double NextDouble(double minValue, double maxValue) => random.NextDouble() * (maxValue - minValue) + minValue;

        public static int Next(int minValue, int maxValue) => random.Next(minValue, maxValue);

        public static double Map(double value, double istart, double istop, double ostart, double ostop) => ostart + (ostop - ostart) * ((value - istart) / (istop - istart));

        public static double[,] NormalizeData(double[,] data)
        {
            double[] min = new double[data.GetLength(1)];
            double[] max = new double[data.GetLength(1)];

            for (int i = 0; i < data.GetLength(0); i++)
                for (int j = 0; j < data.GetLength(1); j++)
                {
                    min[j] = data[i, j] < min[j] ? data[i, j] : min[j];
                    max[j] = data[i, j] > max[j] ? data[i, j] : max[j];
                }

            double[,] normalized = new double[data.GetLength(0), data.GetLength(1)];

            for (int i = 0; i < data.GetLength(0); i++)
                for (int j = 0; j < data.GetLength(1); j++)
                    normalized[i, j] = Map(data[i, j], min[j], max[j], 0, 1);

            return normalized;
        }

        public static double[,] Standardize(double[,] data)
        {
            double cols = data.GetLength(1);

            double[] mean = new double[data.GetLength(1)];
            double[] sd = new double[data.GetLength(1)];

            int rows = data.GetLength(0);

            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    mean[j] += data[i, j] / cols;

            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    sd[j] += (data[i, j] - mean[j]) * (data[i, j] - mean[j]);

            for (int j = 0; j < cols; j++)
                sd[j] /= Math.Sqrt(sd[j] / (rows - 1));

            double[,] z = new double[rows, data.GetLength(1)];

            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    z[i, j] = (data[i, j] - mean[j]) / sd[j];

            return z;
        }

        public static double[] NormalizeVector(double[] data)
        {
            double[] r = new double[data.Length];
            double length = Length(data);

            for (int i = 0; i < data.Length; i++)
                r[i] = data[i] / length;

            return r;
        }

        public static double Length(double[] data)
        {
            double length = 0;

            for (int i = 0; i < data.Length; i++)
                length += data[i] * data[i];

            return Math.Sqrt(length);
        }

        public static double[] GetCentroid(double[,] points)
        {
            double[] r = new double[points.GetLength(1)];

            for (int i = 0; i < points.GetLength(0); i++)
                for (int j = 0; j < points.GetLength(1); j++)
                    r[j] += points[i, j];

            for (int i = 0; i < r.Length; i++)
                r[i] /= points.GetLength(0);

            return r;
        }

        public static double GetDistance(double[] a, double[] b) => Math.Sqrt(GetDistanceSqr(a, b));

        public static double GetDistanceSqr(double[] a, double[] b)
        {
            double distance = 0;

            for (int i = 0; i < a.Length; i++)
                distance += (a[i] - b[i]) * (a[i] - b[i]);

            return distance;
        }

        public static Tuple<Matrix, Matrix, Matrix> QR(double[,] data)
        {
            // A = QR

            Matrix A = new Matrix(data);
            Matrix[] q = new Matrix[data.GetLength(1)];

            q[0] = new Matrix(NormalizeVector(Utility.ExtractColumn(data, 0)));
            //Console.WriteLine($"[{q[0].data[0, 0]}, {q[0].data[1, 0]}, {q[0].data[2, 0]}]");

            for (int i = 1; i < q.Length; i++)
            {
                Matrix column = new Matrix(Utility.ExtractColumn(data, i));

                //Console.WriteLine($"[{column.data[0, 0]}, {column.data[1, 0]}, {column.data[2, 0]}]");

                q[i] = column - ((Matrix.T(column) * q[i - 1]).data[0, 0] * q[i - 1].Normalize()) / q[i - 1].Length();

                for (int j = i - 1; j > 0; j--)
                    q[i] -= ((Matrix.T(column) * q[j].Normalize()).data[0, 0] * q[j].Normalize()) / q[j].Length();
            }

            Matrix Q = new Matrix(q);
            Matrix Qt = Matrix.T(Q);
            Matrix R = Qt * A;

            return new Tuple<Matrix, Matrix, Matrix>(Q, R, Qt);
        }

        // Eigenvalue Power Method
        public static Tuple<double, Matrix> EPM(double[,] data, int steps = 100)
        {
            Matrix m = new Matrix(data);

            // Guess data
            double[] x_0 = new double[m.r];

            for (int i = m.r; i-- > 0;)
                x_0[i] = NextDouble(-1, 1);

            // Guess matrix
            Matrix guess = new Matrix(x_0);

            // lambda_1
            double lambda_1 = -1e9;

            for (int i = steps + 1; i-- > 0;)
            {
                Matrix p = m * guess;
                double n_i = 0;

                for (int r = p.r; r-- > 0;)
                {
                    double abs = Math.Abs(p.data[r, 0]);
                    n_i = abs > n_i ? abs : n_i;
                }

                guess = p / n_i;
                lambda_1 = n_i;
            }

            return new Tuple<double, Matrix>(lambda_1, guess);
        }
    }
}