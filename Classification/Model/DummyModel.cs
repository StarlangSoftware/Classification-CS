using System.Collections.Generic;
using System.IO;
using Classification.Instance;
using Math;

namespace Classification.Model
{
    public class DummyModel : Model
    {
        private readonly DiscreteDistribution _distribution;

        /**
         * <summary> Constructor which sets the distribution using the given {@link InstanceList}.</summary>
         *
         * <param name="trainSet">{@link InstanceList} which is used to get the class distribution.</param>
         */
        public DummyModel(InstanceList.InstanceList trainSet)
        {
            _distribution = trainSet.ClassDistribution();
        }

        /// <summary>
        /// Loads a dummy model from an input model file.
        /// </summary>
        /// <param name="fileName">Model file name.</param>
        public DummyModel(string fileName)
        {
            var input = new StreamReader(fileName);
            _distribution = Model.LoadDiscreteDistribution(input);
            input.Close();
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
    }
}