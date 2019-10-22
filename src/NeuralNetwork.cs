using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Formatters.Binary;

namespace EvoLearning {

    [Serializable]
    public sealed class NeuralNetworkDTO {
        public int[] Layers;
        public SpeciesDTO[] Species;

        public bool IsSame(NeuralNetworkDTO rhs) {
            if (Layers.Length != rhs.Layers.Length) return false;
            for (var i = 0; i < Layers.Length; i++) {
                if (Layers[i] != rhs.Layers[i]) {
                    return false;
                }
            }
            for (var i = 0; i < Species.Length; i++) {
                if (Species[i].Fitness != rhs.Species[i].Fitness) return false;
                for (var j = 0; j < Species[i].Weights.Length; j++) {
                    if (Species[i].Weights[j].IsSame(rhs.Species[i].Weights[j]) == false) {
                        return false;
                    }
                    if (Species[i].Biases[j].IsSame(rhs.Species[i].Biases[j]) == false) {
                        return false;
                    }
                }
            }
            return true;
        }

        public int MaxFitness {
            get {
                var max = 0;
                for (var i = 0; i < Species.Length; i++) {
                    if (Species[i].Fitness > max) {
                        max = Species[i].Fitness;
                    }
                }
                return max;
            }
        }
    }

    [Serializable]
    public sealed class SpeciesDTO {
        public int Fitness;
        public int HistoricMaxFitness;
        public int HistoricStage;
        public int BestStage;
        public int FamilyFactor;
        public TensorDTO[] Weights;
        public TensorDTO[] Biases;
        public int FirstParent;
    }

    public sealed class Species {
        public int Fitness;
        public int HistoricMaxFitness;
        public int HistoricStage;
        public int BestStage;

        public int FamilyFactor;
        public Tensor[] Weights;
        public Tensor[] Biases;
        public int FirstParent;

        public SpeciesDTO ToDTO() {
            var result = new SpeciesDTO();
            result.Fitness = Fitness;
            result.FamilyFactor = FamilyFactor;
            result.Weights = new TensorDTO[Weights.Length];
            for (var i = 0; i < Weights.Length; i++) {
                result.Weights[i] = Weights[i].ToDTO();
            }
            result.Biases = new TensorDTO[Biases.Length];
            for (var i = 0; i < Biases.Length; i++) {
                result.Biases[i] = Biases[i].ToDTO();
            }
            result.FirstParent = FirstParent;
            result.HistoricMaxFitness = HistoricMaxFitness;
            result.BestStage = BestStage;
            result.HistoricStage = HistoricStage;
            return result;
        }

        public static Species FromDTO(SpeciesDTO dto) {
            var result = new Species();
            result.Fitness = dto.Fitness;
            result.FamilyFactor = dto.FamilyFactor;
            result.Weights = new Tensor[dto.Weights.Length];
            for (var i = 0; i < dto.Weights.Length; i++) {
                result.Weights[i] = Tensor.NewFromDTO(dto.Weights[i]);
            }
            result.Biases = new Tensor[dto.Biases.Length];
            for (var i = 0; i < dto.Biases.Length; i++) {
                result.Biases[i] = Tensor.NewFromDTO(dto.Biases[i]);
            }
            result.FirstParent = dto.FirstParent;
            result.HistoricMaxFitness = dto.HistoricMaxFitness;
            result.BestStage = dto.BestStage;
            result.HistoricStage = dto.HistoricStage;
            return result;
        }

        public Species Clone() {
            return FromDTO(ToDTO());
        }
    }

    public enum NeuralNetworkBreedingPhase {
        AverageSum,
        NeuronReplacement
    }

    public sealed class NeuralNetwork {

        const float MaxWeight = 1f;
        const float MinWeight = -1f;
        const float MaxBias = 1f;
        const float MinBias = -1f;

        public static Random _random = new Random();

        int[] _layers = null;
        Tensor _cachedX;

        const int UsingModel = -1;

        NeuralNetworkBreedingPhase _breedingPhase = NeuralNetworkBreedingPhase.AverageSum;

        float[] _mutationSequence = new float[] { 0, 0.0003f, 0.001f, 0.003f, 0.01f, 0.03f, 0.1f, 0.3f};
        int _currentMutationIndex = 0;
        int _warmUpLifeforms;
        int _lifeformsCreated;
        int _speciesCreated;
        List<Species> _speciesStack = null;
        int _speciesMax = 1;
        int _speciesIndex;

        Species _currentSpecimen = null;

        int _maxFitness;
        int _maxSpecies;
        public Tensor[] Activations = null;
        public bool _doingRandom;
        public bool _breeding;


        bool _started = true;
        DateTime _lastWriteTime;

        public NeuralNetwork(int[] layers, Species[] species, int warmUpLifeforms) {
            _cachedX = new Tensor(layers[0], 1);
            _layers = layers;
            _warmUpLifeforms = warmUpLifeforms;
            _maxFitness = -1;
            _maxSpecies = -1;
            for (var i = 0; i < species.Length; i++) {
                if (species[i].Fitness > _maxFitness) {
                    _maxFitness = species[i].Fitness;
                    _maxSpecies = i;
                }
            }
            if (UsingModel >= 0) {
                _speciesIndex = UsingModel;
            }
            Pre.Assert(species.Length > 0, species.Length);
            _speciesStack = new List<Species>(species);
            _currentSpecimen = _speciesStack[_speciesIndex].Clone();
            Activations = new Tensor[layers.Length];
            Logger.LogImportant("Neural Network loaded.");
        }

        public static NeuralNetwork FromScratch(int[] layers) {
            Tensor[] weights;
            Tensor[] bias;
            CreateRandomWeightsAndBiases(layers, out weights, out bias);
            var species = new Species[1] {
                new Species() {
                    Weights = weights,
                    Biases = bias,
                    Fitness = 0,
                    FamilyFactor = 0
                }
            };
            return new NeuralNetwork(layers, species, 2000);
        }

        public static NeuralNetwork FromDTO(NeuralNetworkDTO dto) {
            var species = new Species[dto.Species.Length];
            for (var i = 0; i < species.Length; i++) {
                species[i] = Species.FromDTO(dto.Species[i]);
            }
            return new NeuralNetwork(dto.Layers, species, 0);
        }

        public static NeuralNetworkDTO FromFile(int[] layers) {
            var filename = GetFilename(layers);
            IFormatter formatter = new BinaryFormatter();
            Stream stream = new FileStream(filename, FileMode.Open, FileAccess.Read,  FileShare.Read);
            var dto = (NeuralNetworkDTO) formatter.Deserialize(stream);
            stream.Close();
            return dto;
        }

        public static NeuralNetwork FromFileOrScratch(int[] layers) {
            if (File.Exists(GetFilename(layers))) {
                try {
                    return FromDTO(FromFile(layers));
                } catch (Exception e) {
                    Logger.LogException(e);
                    Logger.LogImportant("Loading network from scratch instead!");
                }
            } 
            return FromScratch(layers);
        }

        public NeuralNetworkDTO ToDTO() {
            var species = new SpeciesDTO[_speciesStack.Count];
            for (var i = 0; i < species.Length; i++) {
                species[i] = _speciesStack[i].ToDTO();
            }
            return new NeuralNetworkDTO() {
                Layers = _layers,
                Species = species
            };
        }

        public void Predict(float[] x_input, ref bool left, ref bool right) {
            Pre.Assert(x_input.Length == _layers[0], x_input.Length, _layers[0]);
            Pre.Assert(_started);

            _cachedX.CopyFrom(x_input);

            var A = _cachedX;
            for (var i = 0; i < _layers.Length - 1; i++)
            {
                Activations[i] = A;
                var Z = _currentSpecimen.Weights[i].Dot(A) + _currentSpecimen.Biases[i];
                if (i == _layers.Length - 1) {
                    A = Z.Apply(Sigmoid);
                } else {
                    A = Z.Apply(Sigmoid);
                }
            }
            Activations[_layers.Length - 1] = A;

            Pre.Assert(A.Shape == "(2, 1)", A.Shape);

            left = A.Get(0, 0) > 0.5;
            right = A.Get(1, 0) > 0.5;

        }

        private void ReloadIfNews() {
            var filename = GetFilename(_layers);
            if (File.Exists(filename) == false) return;

            var date = File.GetLastWriteTime(filename);
            if (date <= _lastWriteTime) return;

            Exception error = null;
            for (var i = 0; i < 10; i++) {
                try {
                    var dto = FromFile(_layers);
                    var otherNN = FromDTO(dto);
                    _maxSpecies = otherNN._maxSpecies;
                    _maxFitness = otherNN._maxFitness;
                    _speciesStack = otherNN._speciesStack;
                    _currentSpecimen = _speciesStack[_speciesIndex].Clone();
                    _lastWriteTime = date;
                    return;
                } catch(Exception e) {
                    error = e;
                }
                for (var j = 0; j < 5; j++) {
                    Logger.LogImportant("Conflict on reading file. " + i);
                }
                System.Threading.Thread.Sleep(1000);
            }
            if (error != null) {
                Logger.LogException(error);
            }
        }


        private void FinishLifeWarmups(int fitness, int stage) {
            _warmUpLifeforms--;
            Logger.LogImportant(_lifeformsCreated + "| Warmups remaining: " + _warmUpLifeforms);
            _currentSpecimen.Fitness = fitness;
            _currentSpecimen.HistoricMaxFitness = fitness;
            _currentSpecimen.FirstParent = _lifeformsCreated;
            _currentSpecimen.BestStage = stage;
            _currentSpecimen.HistoricStage = stage;
            var speciesIndex = -1;
            if (_speciesStack.Count < _speciesMax) {
                Logger.LogImportant(_lifeformsCreated + "| Added Fitness = " + fitness);
                speciesIndex = _speciesStack.Count;
                _speciesStack.Add(_currentSpecimen);
            } else {
                var minSpecimen = -1;
                var minFitness = fitness;
                for (var i = 0; i < _speciesStack.Count; i++) {
                    var candidateSpecimen = _speciesStack[i];
                    if (candidateSpecimen.Fitness < minFitness) {
                        minSpecimen = i;
                        minFitness = candidateSpecimen.Fitness;
                    }
                }
                if (minSpecimen > -1) {
                    Logger.LogImportant(_lifeformsCreated + "| Added Fitness = " + fitness + ", Removed Fitness = " + minFitness + "|" + minSpecimen);
                    _speciesStack[minSpecimen] = _currentSpecimen;
                    speciesIndex = minSpecimen;
                }
            }
            if (fitness > _maxFitness) {
                Pre.Assert(speciesIndex > -1, speciesIndex);
                _maxFitness = fitness;
                _maxSpecies = speciesIndex;
                Logger.LogImportant(_lifeformsCreated + "| NEW MAX GLOBAL!!!");
            } else if (fitness == _maxFitness) {
                Logger.LogImportant(_lifeformsCreated + "| Max global matched!");
            }
            Tensor[] nextW, nextB;
            CreateRandomWeightsAndBiases(_layers, out nextW, out nextB);
            Pre.Assert(nextW != null && nextB != null);
            _currentSpecimen = new Species() {
                Weights = nextW,
                Biases = nextB
            };
        }

        public void FinishLife(int fitness, int stage)
        {

            if (UsingModel >= 0)
            {
                _started = true;
                return;
            }

            if (fitness < 0)
            {
                fitness = 0;
            }

            _lifeformsCreated++;
            if (_lifeformsCreated % 50 == 1)
            {
                if (false && File.Exists(GetFilename(_layers)))
                {
                    if (ToDTO().IsSame(FromFile(_layers)) == false)
                    {
                        Logger.LogWarning("This shouldn't happen!");
                    }
                }
                PrintSpeciesStack();
            }
            ReloadIfNews();
            if (_warmUpLifeforms > 0)
            {
                FinishLifeWarmups(fitness, stage);
                _started = true;
                return;
            }

            var dadSpecimen = _speciesStack[_speciesIndex];
            var mutations = CompareSpecies(dadSpecimen, _currentSpecimen);

            var familyFactor = (int)(fitness * mutations * Math.Log(fitness));
            Pre.Assert(familyFactor >= 0, familyFactor, fitness, mutations);

            var mutationsMessage = (mutations * 100).ToString("0.000") + "%";

            if (fitness > _currentSpecimen.Fitness)
            {
                if (fitness > _maxFitness)
                {
                    Logger.LogImportant(_lifeformsCreated + "| " + mutationsMessage + " Fitness = " + fitness + " NEW GLOBAL MAX!!!!!!!");
                    _maxFitness = fitness;
                }
                else
                {
                    Logger.LogImportant(_lifeformsCreated + "| " + mutationsMessage + " Fitness = " + fitness + " New Specimen Max!!!");
                }
                _currentSpecimen.Fitness = fitness;
                _currentSpecimen.FamilyFactor = familyFactor;
                _currentSpecimen.BestStage = stage;
                dadSpecimen.Biases = _currentSpecimen.Biases;
                dadSpecimen.Weights = _currentSpecimen.Weights;
                dadSpecimen.Fitness = fitness;
                dadSpecimen.FamilyFactor = familyFactor;
                dadSpecimen.BestStage = stage;
                if (fitness > dadSpecimen.HistoricMaxFitness)
                {
                    _currentSpecimen.HistoricMaxFitness = fitness;
                    _currentSpecimen.HistoricStage = stage;
                    dadSpecimen.HistoricMaxFitness = fitness;
                    dadSpecimen.HistoricStage = stage;
                    Logger.LogImportant(_lifeformsCreated + "| LANDMARK IN HISTORY!");
                }

                SaveNetwork();
            }
            else
            {
                if (fitness == _maxFitness)
                {
                    Logger.LogImportant(_lifeformsCreated + "| " + mutationsMessage + " Fitness = " + fitness + ", GLOBAL MAX was matched!!");
                }
                else if (fitness == dadSpecimen.Fitness)
                {
                    Logger.LogImportant(_lifeformsCreated + "| " + mutationsMessage + " Fitness = " + fitness + ", max specimen was matched!!");
                }
                else if (_mutationSequence[_currentMutationIndex] == 0)
                {
                    var oldFitness = dadSpecimen.Fitness + "/" + _maxFitness;
                    dadSpecimen.Fitness -= (dadSpecimen.Fitness - fitness) / 2;
                    Logger.LogImportant(_lifeformsCreated + "| " + mutationsMessage + " Fitness = " + fitness + ", reduced from: " + oldFitness + " to:" + dadSpecimen.Fitness + "/" + _maxFitness + " :(");

                    SaveNetwork();
                }
                else
                {
                    Logger.LogImportant(_lifeformsCreated + "| " + mutationsMessage + " Fitness = " + fitness + ", max value was: " + dadSpecimen.Fitness + "/" + _maxFitness);
                }

                Pre.Assert(_speciesStack.Count == _speciesMax, _speciesStack.Count, _speciesMax);
            }

            _currentSpecimen = CreateNewSpecimen();
            _started = true;
        }

        private Species CreateNewSpecimen()
        {
            _currentMutationIndex++;
            if (_currentMutationIndex >= _mutationSequence.Length)
            {
                _currentMutationIndex = 0;
                _speciesIndex++;
                if (_speciesIndex == _speciesMax)
                {
                    _speciesIndex = 0;
                    switch (_breedingPhase)
                    {
                        case NeuralNetworkBreedingPhase.AverageSum:
                            {
                                _breedingPhase = NeuralNetworkBreedingPhase.NeuronReplacement;
                                break;
                            }
                        case NeuralNetworkBreedingPhase.NeuronReplacement:
                            {
                                _breedingPhase = NeuralNetworkBreedingPhase.AverageSum;
                                break;
                            }
                        default: throw new Exception("Unreachable! " + _breedingPhase);
                    }
                    Logger.LogImportant("# CHANGE BREEDING TO: " + _breedingPhase + "!!!");
                }
                Logger.LogImportant(
                    "# START WITH SPECIMEN: " + _speciesIndex + " -> "
                    + _speciesStack[_speciesIndex].FamilyFactor + "/" + _speciesStack[_speciesIndex].Fitness
                );
            }
            float variance = _mutationSequence[_currentMutationIndex];

            var newSpecimen = _speciesStack[_speciesIndex].Clone();
            switch (_breedingPhase)
            {
                case NeuralNetworkBreedingPhase.AverageSum: {
                    Tensor[] mumW, mumB;
                    CreateRandomWeightsAndBiases(_layers, out mumW, out mumB);
                    for (var i = 1; i < _layers.Length; i++)
                    {
                        newSpecimen.Weights[i - 1] = newSpecimen.Weights[i - 1] * (1 - variance) + mumW[i - 1] * variance;
                        newSpecimen.Biases [i - 1] = newSpecimen.Biases [i - 1] * (1 - variance) + mumB[i - 1] * variance;
                    }
                    break;
                }
                case NeuralNetworkBreedingPhase.NeuronReplacement: {
                    for (var i = 1; i < _layers.Length; i++)
                    {
                        var w = newSpecimen.Weights[i - 1];
                        for (var x = 0; x < w.ShapeX; x++)
                        {
                            for (var y = 0; y < w.ShapeY; y++)
                            {
                                if (_random.NextDouble() > variance) continue;
                                w.Set(x, y, RandomWeight(_random));
                            }
                        }
                        var b = newSpecimen.Biases[i - 1];
                        for (var x = 0; x < b.ShapeX; x++)
                        {
                            for (var y = 0; y < b.ShapeY; y++)
                            {
                                if (_random.NextDouble() > variance) continue;
                                b.Set(x, y, RandomBias(_random));
                            }
                        }
                    }
                    break;
                }
                default: throw new Exception("Unreachable! " + _breedingPhase);
            }
            return newSpecimen;
        }

        private double CompareSpecies(Species lhs, Species rhs)
        {
            var lhsW = lhs.Weights;
            var lhsB = lhs.Biases;
            var rhsW = rhs.Weights;
            var rhsB = rhs.Biases;
            var mutations = 0.0;
            for (var i = 0; i < lhsB.Length; i++)
            {
                mutations += rhsB[i].Compare(lhsB[i]).SumAll() / lhsB[i].TotalCells / (MaxBias - MinBias);
                mutations += rhsW[i].Compare(lhsW[i]).SumAll() / lhsW[i].TotalCells / (MaxWeight - MinWeight);
            }
            mutations /= (lhsB.Length + lhsW.Length) / 2;
            return mutations;
        }

        public static bool IsFileReady(String sFilename)
        {
            // If the file can be opened for exclusive access it means that the file
            // is no longer locked by another process.
            try
            {
                using (FileStream inputStream = File.Open(sFilename, FileMode.Open, FileAccess.Read, FileShare.None))
                {
                    if (inputStream.Length > 0)
                    {
                        return true;
                    }
                    else
                    {
                        return false;
                    }

                }
            }
            catch (Exception)
            {
                return false;
            }
        }

        private Exception WriteFile(string filename) {
            Exception error = null;
            try
            {
                IFormatter formatter = new BinaryFormatter();
                var stream = new FileStream(filename, FileMode.Create, FileAccess.Write, FileShare.ReadWrite);
                formatter.Serialize(stream, ToDTO());
                stream.Close();
                _lastWriteTime = File.GetLastWriteTime(filename);
                return null;
            }
            catch (Exception e)
            {
                error = e;
            }
            return error;
        }

        private void SaveNetwork()
        {
            Exception error = null;
            var filename = GetFilename(_layers);
            for (var i = 0; i < 10; i++) {
                error = WriteFile(filename);
                if (error == null) {
                    return;
                }
                for (var j = 0; j < 5; j++) {
                    Logger.LogImportant("Conflict on writting file. " + i);
                }
                System.Threading.Thread.Sleep(1000);
            }
            Logger.LogException(error);
        }

        private static float RandomBias(Random random)
        {
            return (float)(random.NextDouble() * (MaxBias - MinBias)) - MinBias;
        }

        private static float RandomWeight(Random random)
        {
            return (float)(random.NextDouble() * (MaxWeight - MinWeight)) - MinWeight;
        }

        private static void CreateRandomWeightsAndBiases(int[] layers, out Tensor[] mumW, out Tensor[] mumB)
        {
            mumW = new Tensor[layers.Length - 1];
            mumB = new Tensor[layers.Length - 1];
            for (var i = 1; i < layers.Length; i++)
            {
                mumW[i - 1] = (Tensor.NewFromRandom(layers[i], layers[i - 1], _random) * (MaxWeight - MinWeight)) + MinWeight;
                mumB[i - 1] = (Tensor.NewFromRandom(layers[i], 1, _random) * (MaxBias - MinBias)) + MinBias;
            }
        }

        public void PrintWeightsAndBiases() {
            var message = "WEIGHTS AND BIASES";
            for (var i = 0; i < _layers.Length - 1; i++) {
                message += "\nLAYER: " + (i + 1);
                message += "\nWEIGHTS: " + _currentSpecimen.Weights[i];
                message += "\nBIASES: " + _currentSpecimen.Biases[i];
            }
            Logger.LogImportant(message);
        }

        public void PrintSpeciesStack() {
            var message = "Species Stack: \n";
            for (var i = 0; i < _speciesStack.Count; i++) {
                var specimen = _speciesStack[i];
                message += i + "| " + specimen.FirstParent + " ... " + specimen.FamilyFactor 
                    + " -> " + specimen.Fitness + "-" + specimen.BestStage 
                    + "/" + specimen.HistoricMaxFitness + "-" + specimen.HistoricStage + "\n";
            }
            message += "Compares: ";
            var mutationMatrix = new double[_speciesStack.Count][];
            for (var i = 0; i < _speciesStack.Count; i++) {
                mutationMatrix[i] = new double[_speciesStack.Count];
                message += "       " + i + " #";
            }
            message += "\n";
            for (var i = 0; i < _speciesStack.Count; i++) {
                message += "       " + i + " # ";
                for (var j = 0; j < _speciesStack.Count; j++) {
                    if (j == i) {
                        message += "          ";
                        continue;
                    }
                    double compareValue;
                    if (mutationMatrix[j][i] != 0) {
                        compareValue = mutationMatrix[j][i];
                    } else {
                        compareValue = CompareSpecies(_speciesStack[i], _speciesStack[j]) * 100;
                    }
                    mutationMatrix[i][j] = compareValue;
                    message += compareValue.ToString("00.00000") + "% ";
                }
                message += "\n";
            }
            Logger.LogImportant(message);
        }

        private static Tensor[] Clone(Tensor[] o) {
            var result = new Tensor[o.Length];
            for (var i = 0; i < o.Length; i++) {
                result[i] = o[i].Clone();
            }
            return result;
        }

        private static string GetFilename(int[] layers) {
            var filename = "nn_species_";
            for (var i = 0; i < layers.Length; i++) {
                filename += layers[i];
                filename += "_";
            }
            filename += ".bin";
            return filename;
        }

        public void NextLife() {
            _started = true;
        }

        public static float Sigmoid(float n)
        {
            return (float)(1.0 / (1.0 + Math.Exp(-n)));
        }

        public static float ReLU(float n) {
            var result = Math.Max(0, n);
            return result;
        }

        public static float Tanh(float n) {
            var result = 2 * Sigmoid(2 * n) - 1;
            return result;
        }
    }
}