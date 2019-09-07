using System;

namespace SNNP.MLP
{
    public class Matrix
    {
        public int r, c;

        public double[,] data;

        public Matrix(int rows, int columns, bool randomize = false)
        {
            r = rows;
            c = columns;

            data = new double[r, c];

            if (randomize)
                Randomize();
        }

        public Matrix(double[] inputs)
        {
            r = inputs.Length;
            c = 1;

            data = new double[r, c];

            for (int i = 0; i < r; i++)
                data[i, 0] = inputs[i];
        }

        public void Randomize()
        {
            for (int i = 0; i < r; i++)
                for (int j = 0; j < c; j++)
                    data[i, j] = Utility.NextDouble(-0.5, 0.5);
        }

        public void T()
        {
            int rows = c;
            int columns = r;

            double[,] newData = new double[rows, columns];

            for (int i = 0; i < r; i++)
                for (int j = 0; j < c; j++)
                    newData[j, i] = data[i, j];

            r = rows;
            c = columns;
            data = newData;
        }

        public double[] ToArray()
        {
            double[] ret = new double[r * c];

            for (int i = 0; i < r; i++)
                for (int j = 0; j < c; j++)
                    ret[i + j * c] = data[i, j];

            return ret;
        }

        public double SquaredSum()
        {
            double ret = 0;

            for (int i = 0; i < r; i++)
                    ret += data[i, 0] * data[i, 0];

            return ret / (double)r;
        }

        public void Map(Func<double, double> f)
        {
            for (int i = 0; i < r; i++)
                for (int j = 0; j < c; j++)
                    data[i, j] = f(data[i, j]);
        }

        public static Matrix Map(Matrix m, Func<double, double> f)
        {
            for (int i = 0; i < m.r; i++)
                for (int j = 0; j < m.c; j++)
                    m.data[i, j] = f(m.data[i, j]);

            return m;
        }

        public static Matrix T(Matrix m)
        {
            Matrix ret = new Matrix(m.c, m.r);

            for (int i = 0; i < m.r; i++)
                for (int j = 0; j < m.c; j++)
                    ret.data[j, i] = m.data[i, j];

            return ret;
        }

        public static Matrix operator +(Matrix a, Matrix b)
        {
            Matrix ret = new Matrix(a.r, a.c);

            for (int i = 0; i < b.r; i++)
                for (int j = 0; j < b.c; j++)
                    ret.data[i, j] = a.data[i, j] + b.data[i, j];

            return ret;
        }

        public static Matrix operator -(Matrix a, Matrix b)
        {
            Matrix ret = new Matrix(a.r, a.c);

            for (int i = 0; i < b.r; i++)
                for (int j = 0; j < b.c; j++)
                    ret.data[i, j] = a.data[i, j] - b.data[i, j];

            return ret;
        }

        public static Matrix operator *(Matrix a, Matrix b)
        {
            if (a.c != b.r)
                throw new System.NotImplementedException($"Linhas de a = {a.r} diferente de colunas de b = {b.c}");
            Matrix ret = new Matrix(a.r, b.c);

            for (int i = 0; i < ret.r; i++)
                for (int j = 0; j < ret.c; j++)
                {
                    double sum = 0;
                    for (int k = 0; k < b.r; k++) // a.columns == b.rows
                        sum += a.data[i, k] * b.data[k, j];
                    ret.data[i, j] = sum;
                }

            return ret;
        }

public static Matrix operator *(Matrix m, double n)
{
    Matrix ret = new Matrix(m.r, m.c);

    for (int i = 0; i < m.r; i++)
        for (int j = 0; j < m.c; j++)
            ret.data[i, j] = m.data[i, j] * n;

    return ret;
}

        public static Matrix operator ^(Matrix m, double n)
        {
            for (int i = 0; i < m.r; i++)
                for (int j = 0; j < m.c; j++)
                    m.data[i, j] = Math.Pow(m.data[i, j], n);

            return m;
        }

        public static Matrix operator %(Matrix a, Matrix b)
        {
            for (int i = 0; i < b.r; i++)
                for (int j = 0; j < b.c; j++)
                    a.data[i, j] = a.data[i, j] * b.data[i, j];

            return a;
        }

public static Matrix operator *(double n, Matrix m)
{
    return m * n;
}
    }
}