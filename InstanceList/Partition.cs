using System.Collections.Generic;

namespace Classification.InstanceList
{
    public class Partition
    {
        private readonly List<InstanceList> _multiList;

        /**
         * <summary> Constructor for generating a partition.</summary>
         */
        public Partition()
        {
            _multiList = new List<InstanceList>();
        }

        /**
         * <summary> Adds given instance list to the list of instance lists.</summary>
         *
         * <param name="list">Instance list to add.</param>
         */
        public void Add(InstanceList list)
        {
            _multiList.Add(list);
        }

        /**
         * <summary> Returns the size of the list of instance lists.</summary>
         *
         * <returns>The size of the list of instance lists.</returns>
         */
        public int Size()
        {
            return _multiList.Count;
        }

        /**
         * <summary> Returns the corresponding instance list at given index of list of instance lists.</summary>
         *
         * <param name="index">Index of the instance list.</param>
         * <returns>Instance list at given index of list of instance lists.</returns>
         */
        public InstanceList Get(int index)
        {
            return _multiList[index];
        }

        /**
         * <summary> Returns the instances of the items at the list of instance lists.</summary>
         *
         * <returns>Instances of the items at the list of instance lists.</returns>
         */
        public List<Instance.Instance>[] GetLists()
        {
            var result = new List<Instance.Instance>[_multiList.Count];
            for (var i = 0; i < _multiList.Count; i++)
            {
                result[i] = _multiList[i].GetInstances();
            }

            return result;
        }
    }
}