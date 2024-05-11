using System.Collections.Generic;
using System.IO;
using Classification.Instance;

namespace Classification.Model.DecisionTree
{
    public class DecisionTree : ValidatedModel
    {
        private readonly DecisionNode _root;

        /**
         * <summary> Constructor that sets root node of the decision tree.</summary>
         *
         * <param name="root">DecisionNode type input.</param>
         */
        public DecisionTree(DecisionNode root)
        {
            _root = root;
        }

        /// <summary>
        /// Loads a decision tree model from an input model file.
        /// </summary>
        /// <param name="fileName">Model file name.</param>
        public DecisionTree(string fileName)
        {
            var input = new StreamReader(fileName);
            _root = new DecisionNode(input);
            input.Close();
        }

        /**
         * <summary> The predict method  performs prediction on the root node of given instance, and if it is null, it returns the possible class labels.
         * Otherwise it returns the returned class labels.</summary>
         *
         * <param name="instance">Instance make prediction.</param>
         * <returns>Possible class labels.</returns>
         */
        public override string Predict(Instance.Instance instance)
        {
            var predictedClass = _root.Predict(instance);
            if (predictedClass == null && instance is CompositeInstance) {
                predictedClass = ((CompositeInstance) instance).GetPossibleClassLabels()[0];
            }
            return predictedClass;
        }

        /// <summary>
        /// Calculates the posterior probability distribution for the given instance according to Decision tree model.
        /// </summary>
        /// <param name="instance">Instance for which posterior probability distribution is calculated.</param>
        /// <returns>Posterior probability distribution for the given instance.</returns>
        public override Dictionary<string, double> PredictProbability(Instance.Instance instance)
        {
            return _root.PredictProbabilityDistribution(instance);
        }

        /**
         * <summary> The prune method takes an {@link InstanceList} and  performs pruning to the root node.</summary>
         *
         * <param name="pruneSet">{@link InstanceList} to perform pruning.</param>
         */
        public void Prune(InstanceList.InstanceList pruneSet)
        {
            _root.Prune(this, pruneSet);
        }
    }
}