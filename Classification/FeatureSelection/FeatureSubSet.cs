using System;
using System.Collections.Generic;

namespace Classification.FeatureSelection
{
    public class FeatureSubSet : ICloneable
    {
        private readonly List<int> _indexList;

        /**
         * <summary> A constructor that sets the indexList {@link List}.</summary>
         *
         * <param name="indexList">An List consists of integer indices.</param>
         */
        public FeatureSubSet(List<int> indexList)
        {
            this._indexList = indexList;
        }

        /**
         * <summary> A constructor that takes number of features as input and initializes indexList with these numbers.</summary>
         *
         * <param name="numberOfFeatures">Indicates the indices of indexList.</param>
         */
        public FeatureSubSet(int numberOfFeatures)
        {
            _indexList = new List<int>();
            for (var i = 0; i < numberOfFeatures; i++)
            {
                _indexList.Add(i);
            }
        }

        /**
         * <summary> A constructor that creates a new List for indexList.</summary>
         */
        public FeatureSubSet()
        {
            _indexList = new List<int>();
        }

        /**
         * <summary> The clone method creates a new List with the elements of indexList and returns it as a new FeatureSubSet.</summary>
         *
         * <returns>A new List with the elements of indexList and returns it as a new FeatureSubSet.</returns>
         */
        public object Clone()
        {
            var newIndexList = new List<int>();
            foreach (var index in _indexList) {
                newIndexList.Add(index);
            }
            return new FeatureSubSet(newIndexList);
        }

        /**
         * <summary> The size method returns the size of the indexList.</summary>
         *
         * <returns>The size of the indexList.</returns>
         */
        public int Size()
        {
            return _indexList.Count;
        }

        /**
         * <summary> The get method returns the item of indexList at given index.</summary>
         *
         * <param name="index">Index of the indexList to be accessed.</param>
         * <returns>The item of indexList at given index.</returns>
         */
        public int Get(int index)
        {
            return _indexList[index];
        }

        /**
         * <summary> The contains method returns True, if indexList contains given input number and False otherwise.</summary>
         *
         * <param name="featureNo">Feature number that will be checked.</param>
         * <returns>True, if indexList contains given input number.</returns>
         */
        public bool Contains(int featureNo)
        {
            return _indexList.Contains(featureNo);
        }

        /**
         * <summary> The add method adds given int to the indexList.</summary>
         *
         * <param name="featureNo">int that will be added to indexList.</param>
         */
        public void Add(int featureNo)
        {
            _indexList.Add(featureNo);
        }

        /**
         * <summary> The remove method removes the item of indexList at the given index.</summary>
         *
         * <param name="index">Index of the item that will be removed.</param>
         */
        public void Remove(int index)
        {
            _indexList.RemoveAt(index);
        }
    }
}