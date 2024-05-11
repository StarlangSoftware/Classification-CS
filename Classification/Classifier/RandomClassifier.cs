using System;
using System.Collections.Generic;
using Classification.Model;

namespace Classification.Classifier
{
    public class RandomClassifier : Classifier
    {
        /**
         * <summary> Training algorithm for random classifier.</summary>
         *
         * <param name="trainSet">  Training data given to the algorithm.</param>
         * <param name="parameters">-</param>
         */
        public override void Train(InstanceList.InstanceList trainSet, Parameter.Parameter parameters) {
            model = new RandomModel(new List<String>(trainSet.ClassDistribution().Keys), parameters.GetSeed());
        }

        /// <summary>
        /// Loads the random classifier model from an input file.
        /// </summary>
        /// <param name="fileName">File name of the random classifier model.</param>
        public override void LoadModel(string fileName)
        {
            model = new RandomModel(fileName);
        }
    }
}