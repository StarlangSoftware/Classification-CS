using System.Collections.Generic;
using Classification.Experiment;
using Classification.Performance;

namespace Classification.FeatureSelection
{
    public abstract class SubSetSelection
    {
        protected FeatureSubSet initialSubSet;

        protected abstract List<FeatureSubSet> Operator(FeatureSubSet current, int numberOfFeatures);

        /**
         * <summary> A constructor that sets the initial subset with given input.</summary>
         *
         * <param name="initialSubSet">{@link FeatureSubSet} input.</param>
         */
        public SubSetSelection(FeatureSubSet initialSubSet)
        {
            this.initialSubSet = initialSubSet;
        }

        /**
         * <summary> The forward method starts with having no feature in the model. In each iteration, it keeps adding the features that are not currently listed.</summary>
         *
         * <param name="currentSubSetList">List to add the FeatureSubsets.</param>
         * <param name="current">          FeatureSubset that will be added to currentSubSetList.</param>
         * <param name="numberOfFeatures"> The number of features to add the subset.</param>
         */
        protected void Forward(List<FeatureSubSet> currentSubSetList, FeatureSubSet current, int numberOfFeatures)
        {
            for (var i = 0; i < numberOfFeatures; i++)
            {
                if (!current.Contains(i))
                {
                    var candidate = (FeatureSubSet) current.Clone();
                    candidate.Add(i);
                    currentSubSetList.Add(candidate);
                }
            }
        }

        /**
         * <summary> The backward method starts with all the features and removes the least significant feature at each iteration.</summary>
         *
         * <param name="currentSubSetList">List to add the FeatureSubsets.</param>
         * <param name="current">          FeatureSubset that will be added to currentSubSetList</param>
         */
        protected void Backward(List<FeatureSubSet> currentSubSetList, FeatureSubSet current)
        {
            for (var i = 0; i < current.Size(); i++)
            {
                var candidate = (FeatureSubSet) current.Clone();
                candidate.Remove(i);
                currentSubSetList.Add(candidate);
            }
        }

        /**
         * <summary> The execute method takes an {@link Experiment} and a {@link MultipleRun} as inputs. By selecting a candidateList from given
         * Experiment it tries to find a FeatureSubSet that gives best performance.</summary>
         *
         * <param name="multipleRun">{@link MultipleRun} type input.</param>
         * <param name="experiment"> {@link Experiment} type input.</param>
         * <returns>FeatureSubSet that gives best performance.</returns>
         */
        public FeatureSubSet Execute(MultipleRun multipleRun, Experiment.Experiment experiment)
        {
            var processed = new HashSet<FeatureSubSet>();
            var best = initialSubSet;
            processed.Add(best);
            var betterFound = true;
            ExperimentPerformance bestPerformance = null;
            if (best.Size() > 0)
            {
                bestPerformance = multipleRun.Execute(experiment.FeatureSelectedExperiment(best));
            }

            while (betterFound)
            {
                betterFound = false;
                var candidateList = Operator(best,
                    experiment.GetDataSet().GetDataDefinition().AttributeCount());
                foreach (var candidateSubSet in candidateList)
                {
                    if (!processed.Contains(candidateSubSet))
                    {
                        if (candidateSubSet.Size() > 0)
                        {
                            var currentPerformance =
                                multipleRun.Execute(experiment.FeatureSelectedExperiment(candidateSubSet));
                            if (bestPerformance == null || currentPerformance.IsBetter(bestPerformance))
                            {
                                best = candidateSubSet;
                                bestPerformance = currentPerformance;
                                betterFound = true;
                            }
                        }

                        processed.Add(candidateSubSet);
                    }
                }
            }

            return best;
        }
    }
}