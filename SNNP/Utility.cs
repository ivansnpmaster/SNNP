using System;
using System.Collections.Generic;
using Microsoft.VisualBasic.FileIO;

namespace SNNP.MLP
{
    public static class Utility
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

        public static double[,] LoadCsv(string path, string delimiter = ",", string commentToken = "#", bool quotes = false, bool header = false)
        {
            Console.WriteLine(string.Format("Loading file at {0}", path));

            using (TextFieldParser csvParser = new TextFieldParser(path))
            {
                csvParser.CommentTokens = new string[] { commentToken };
                csvParser.SetDelimiters(new string[] { delimiter });
                csvParser.HasFieldsEnclosedInQuotes = quotes;

                if (header)
                    csvParser.ReadLine();

                List<double[]> data = new List<double[]>();

                while (!csvParser.EndOfData)
                {
                    // Read current line fields, pointer moves to the next line.
                    string[] fields = csvParser.ReadFields();
                    data.Add(new double[fields.Length]);

                    for (int i = 0; i < fields.Length; i++)
                        data[data.Count - 1][i] = Convert.ToDouble(fields[i]);
                }

                double[,] newData = new double[data.Count, data[0].Length];

                for (int i = 0; i < newData.GetLength(0); i++)
                    for (int j = 0; j < newData.GetLength(1); j++)
                        newData[i, j] = data[i][j];

                Console.WriteLine("End of loading the file");

                return newData;
            }
        }

        public static double[] GetCentroid(double[,] points)
        {
            double[] r = new double[points.GetLength(1)];

            for (int i = 0; i < points.GetLength(0); i++)
                for (int j = 0; j < points.GetLength(1); j++)
                    r[i] += points[i, j];

            for (int i = 0; i < r.Length; i++)
                r[i] /= points.GetLength(0);

            return r;
        }

        public static double GetDistance(double[] a, double[] b)
        {
            double distance = 0;

            for (int i = 0; i < a.Length; i++)
                distance += (a[i] - b[i]) * (a[i] - b[i]);
            
            return Math.Sqrt(distance);
        }
    }
}