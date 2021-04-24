using System.Collections.Generic;
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
            this._root = root;
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