using System.Collections.Generic;
using System.IO;
using Classification.Instance;
using Math;

namespace Classification.Model
{
    public class DummyModel : Model
    {
        private DiscreteDistribution _distribution;

        /**
         * <summary> Constructor which sets the distribution using the given {@link InstanceList}.</summary>
         */
        public DummyModel()
        {
        }

        /// <summary>
        /// Loads a dummy model from an input model file.
        /// </summary>
        /// <param name="fileName">Model file name.</param>
        private void Load(string fileName)
        {
            var input = new StreamReader(fileName);
            _distribution = Model.LoadDiscreteDistribution(input);
            input.Close();
        }
        
        /// <summary>
        /// Loads a dummy model from an input model file.
        /// </summary>
        /// <param name="fileName">Model file name.</param>
        public DummyModel(string fileName)
        {
            Load(fileName);
        }

        /**
         * <summary> The predict method takes an Instance as an input and returns the entry of distribution which has the maximum value.</summary>
         *
         * <param name="instance">Instance to make prediction.</param>
         * <returns>The entry of distribution which has the maximum value.</returns>
         */
        public override string Predict(Instance.Instance instance)
        {
            if (instance is CompositeInstance compositeInstance)
            {
                var possibleClassLabels = compositeInstance.GetPossibleClassLabels();
                return _distribution.GetMaxItem(possibleClassLabels);
            }

            return _distribution.GetMaxItem();
        }

        /// <summary>
        /// Calculates the posterior probability distribution for the given instance according to dummy model.
        /// </summary>
        /// <param name="instance">Instance for which posterior probability distribution is calculated.</param>
        /// <returns>Posterior probability distribution for the given instance.</returns>
        public override Dictionary<string, double> PredictProbability(Instance.Instance instance)
        {
            return _distribution.GetProbabilityDistribution();
        }

        /**
        * <summary> Training algorithm for the dummy classifier. Actually dummy classifier returns the maximum occurring class in
        * the training data, there is no training.</summary>
        *
        * <param name="trainSet">  Training data given to the algorithm.</param>
        * <param name="parameters">-</param>
        */
        public override void Train(InstanceList.InstanceList trainSet, Parameter.Parameter parameters)
        {
            _distribution = trainSet.ClassDistribution();
        }

        /// <summary>
        /// Loads the dummy model from an input file.
        /// </summary>
        /// <param name="fileName">File name of the dummy model.</param>
        public override void LoadModel(string fileName)
        {
            Load(fileName);
        }
    }
}