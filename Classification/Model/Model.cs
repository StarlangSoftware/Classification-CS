using System;
using System.Collections.Generic;
using System.IO;
using Math;

namespace Classification.Model
{
    public abstract class Model
    {
        /**
         * <summary> An abstract predict method that takes an {@link Instance} as an input.</summary>
         *
         * <param name="instance">{@link Instance} to make prediction.</param>
         * <returns>The class label as a String.</returns>
         */
        public abstract string Predict(Instance.Instance instance);

        public abstract Dictionary<string, double> PredictProbability(Instance.Instance instance);

        protected InstanceList.InstanceList LoadInstanceList(StreamReader input)
        {
            var types = input.ReadLine().Split(" ");
            var instanceCount = int.Parse(input.ReadLine());
            var instanceList = new InstanceList.InstanceList();
            for (var i = 0; i < instanceCount; i++)
            {
                instanceList.Add(LoadInstance(input.ReadLine(), types));
            }

            return instanceList;
        }

        protected Instance.Instance LoadInstance(string line, string[] attributeTypes)
        {
            var items = line.Split(",");
            var instance = new Instance.Instance(items[items.Length - 1]);
            for (var i = 0; i < items.Length - 1; i++)
            {
                switch (attributeTypes[i])
                {
                    case "DISCRETE":
                        instance.AddAttribute(items[i]);
                        break;
                    case "CONTINUOUS":
                        instance.AddAttribute(double.Parse(items[i]));
                        break;
                }
            }

            return instance;
        }

        protected Matrix LoadMatrix(StreamReader input)
        {
            var items = input.ReadLine().Split(" ");
            var matrix = new Matrix(int.Parse(items[0]), int.Parse(items[1]));
            for (var j = 0; j < matrix.GetRow(); j++){
                var line = input.ReadLine();
                items = line.Split(" ");
                for (var k = 0; k < matrix.GetColumn(); k++){
                    matrix.SetValue(j, k, Double.Parse(items[k]));
                }
            }
            return matrix;
        }

    }
}