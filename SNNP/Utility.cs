using System;
using System.Data;
using System.Collections.Generic;

namespace SNNP
{
    public static class Utility
    {
        public static double[,] LoadCsv(string path, string delimiter = ",", string commentToken = "#", bool quotes = false, bool header = false)
        {
            using (var csvParser = new Microsoft.VisualBasic.FileIO.TextFieldParser(path))
            {
                csvParser.CommentTokens = new string[] { commentToken };
                csvParser.SetDelimiters(new string[] { delimiter });
                csvParser.HasFieldsEnclosedInQuotes = quotes;

                if (header)
                    csvParser.ReadLine();

                var data = new List<double[]>();

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

                return newData;
            }
        }

        public static double[] ExtractColumn(double[,] data, int columnIndex)
        {
            double[] r = new double[data.GetLength(0)];

            for (int i = r.Length; i-- > 0;)
                r[i] = data[i, columnIndex];

            return r;
        }

        /// <summary>
        /// Cast a DataTable to a double[,]. All the data in the DataTable must be numbers.
        /// </summary>
        /// <param name="datatable">DataTable to be casted to a double[,].</param>
        /// <returns>Returns the casted DataTable as a double[,].</returns>
        public static double[,] CastDataTable(DataTable datatable)
        {
            try
            {
                int r = datatable.Rows.Count;
                int c = datatable.Columns.Count;
                double[,] data = new double[r, c];

                for (int i = 0; i < r; i++)
                    for (int j = 0; j < c; j++)
                        data[i, j] = Convert.ToDouble(datatable.Rows[i][j]);

                return data;
            }
            catch (Exception) { throw; }
        }

        public static void ExportCSV(object[,] exportData, string filePath = null)
        {
            var file = string.Format(@"{0}\{1}.csv", filePath ?? AppContext.BaseDirectory, DateTime.Now.ToLongDateString());

            using (var stream = System.IO.File.CreateText(file))
            {
                int rows = exportData.GetLength(0);
                int cols = exportData.GetLength(1);

                for (int i = 0; i < rows; i++)
                {
                    string csvRow = "";

                    for (int j = 0; j < cols - 1; j++)
                        csvRow += string.Format("{0};", exportData[i, j].ToString());

                    csvRow += exportData[i, cols - 1].ToString();

                    stream.WriteLine(csvRow);
                }
            }
        }

        public static bool Save<T>(string filePath, object objectToSave)
        {
            try
            {
                using (var stream = System.IO.File.Open(string.Format("{0}.bin", filePath), System.IO.FileMode.Create))
                {
                    var bin = new System.Runtime.Serialization.Formatters.Binary.BinaryFormatter();
                    bin.Serialize(stream, objectToSave);
                }

                return true;
            }
            catch (System.IO.IOException) { throw; }
        }

        public static T Load<T>(string filePath)
        {
            try
            {
                object o;

                using (var stream = System.IO.File.Open(filePath, System.IO.FileMode.Open))
                {
                    var bin = new System.Runtime.Serialization.Formatters.Binary.BinaryFormatter();
                    o = (T)bin.Deserialize(stream);
                }

                return (T)o;
            }
            catch (System.IO.IOException) { throw; }
        }
    }
}