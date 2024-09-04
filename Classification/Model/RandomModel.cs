using System;
using System.Collections.Generic;
using System.IO;
using Classification.Instance;

namespace Classification.Model
{
    public class RandomModel : Model
    {
        private List<string> _classLabels;
        private Random _random;
        private int _seed;

        public RandomModel()
        {
        }

        /// <summary>
        /// Loads a random classifier model from an input model file.
        /// </summary>
        /// <param name="fileName">Model file name.</param>
        private void Load(string fileName)
        {
            var input = new StreamReader(fileName);
            _seed = int.Parse(input.ReadLine());
            _random = new Random(_seed);
            var size = int.Parse(input.ReadLine());
            _classLabels = new List<string>();
            for (var i = 0; i < size; i++){
                _classLabels.Add(input.ReadLine());
            }
            input.Close();
        }
        
        /// <summary>
        /// Loads a random classifier model from an input model file.
        /// </summary>
        /// <param name="fileName">Model file name.</param>
        public RandomModel(string fileName)
        {
            Load(fileName);
        }
        
        /**
         * <summary> The predict method gets an Instance as an input and retrieves the possible class labels as an List. Then selects a
         * random number as an index and returns the class label at this selected index.</summary>
         *
         * <param name="instance">{@link Instance} to make prediction.</param>
         * <returns>The class label at the randomly selected index.</returns>
         */
        public override string Predict(Instance.Instance instance)
        {
            if (instance is CompositeInstance compositeInstance) {
                var possibleClassLabels = compositeInstance.GetPossibleClassLabels();
                var size = possibleClassLabels.Count;
                var index = _random.Next(size);
                return possibleClassLabels[index];
            } else {
                var size = _classLabels.Count;
                var index = _random.Next(size);
                return _classLabels[index];
            }
        }

        /// <summary>
        /// Calculates the posterior probability distribution for the given instance according to random model.
        /// </summary>
        /// <param name="instance">Instance for which posterior probability distribution is calculated.</param>
        /// <returns>Posterior probability distribution for the given instance.</returns>
        public override Dictionary<string, double> PredictProbability(Instance.Instance instance)
        {
            var result = new Dictionary<string, double>();
            foreach (string classLabel in _classLabels)
            {
                result[classLabel] = 1.0 / _classLabels.Count;
            }

            return result;
        }
        
        /**
         * <summary> Training algorithm for random classifier.</summary>
         *
         * <param name="trainSet">  Training data given to the algorithm.</param>
         * <param name="parameters">-</param>
         */
        public override void Train(InstanceList.InstanceList trainSet, Parameter.Parameter parameters) {
            _classLabels = new List<string>(trainSet.ClassDistribution().Keys);
            _seed = parameters.GetSeed();
            _random = new Random(_seed);
        }

        /// <summary>
        /// Loads the random classifier model from an input file.
        /// </summary>
        /// <param name="fileName">File name of the random classifier model.</param>
        public override void LoadModel(string fileName)
        {
            Load(fileName);
        }

    }
}