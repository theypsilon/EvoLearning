# EvoLearning
Simple Tensor, Neural Network and Evolutionary Reinforcement Learning libraries

## Bot Example

```csharp
public sealed class NeuralNetworkBot : IInputWriter {
    private NeuralNetwork _nn = null;

    private Context _ctx; // This is a struct containing the simulation state, defined elsewhere.
    double _fitness = 0;
    const int InputRaysSize = 30;
    float[] _inputRays = new float[InputRaysSize];

    public NeuralNetworkBot(Context ctx) {
        _ctx = ctx;
        _nn = NeuralNetwork.FromFileOrScratch(new int[] {InputRaysSize, 500, 2});
        Pre.Assert(ctx.State.Config.DLRays > 0, ctx.State.Config.DLRays);
    }

    public void WriteInput(ref Input input) {

        if (_ctx.State.Player.HasJustDied) {
            _nn.FinishLife((int) _fitness, (int) _ctx.State.Stage.StageNumber);
        } else if (_ctx.State.Player.HasJustReborn) {
            _nn.NextLife();
            _fitness = 0;
        }

        if (_ctx.State.Player.IsDead) {
            return;
        }

        // These components are using the following library: https://github.com/theypsilon/MiniECS
        var playerPosition = _ctx.Pools.PositionComponents.Get(_ctx.State.Player.Entity);

        var fitnessChange = 0.0;
        float[] inputNeurons = null;

        // The input neurons are values between [0.0, 1.0] indicating the proximity of an enemy
        // 0 indicates undetected enemy
        // 0.5 indicates enemy detected at intermediate distance
        // 1.0 indicates enemy at collision distance
        // It basically works like a sonar, with rays pointing all coordinates around
        
        var neurons = _ctx.State.DeepLearning.Rays; 
        if (neurons == null) return;

        for (var i = 0; i < neurons.Length; i++) {
            _inputRays[i] = neurons[i];
            if (neurons[i] >= 0.85) { // indicates an enemy is very close, I'd like to punish that.
                fitnessChange = -1.5;
            }
        }
        for (var i = 0; i < neurons.Length; i++) {
            _inputRays[15 + i] = playerPosition.Point.X / 1080; // Player position is between [0.0, 1080.0]
        }
        inputNeurons = _inputRays;
        
        _nn.Predict(inputNeurons, ref input.Control.Left, ref input.Control.Right);

        // rewarding survivavility
        fitnessChange += 1.0;
        
        // rewarding not moving
        if (input.Control.Left == false) {
            fitnessChange += 0.1;
        }
        if (input.Control.Right == false) {
            fitnessChange += 0.1;
        }
        
        // rewarding staying in the center of the screen
        if (playerPosition.Point.X < 100) {
            fitnessChange -= 0.25;
        }
        if (playerPosition.Point.X > 980) {
            fitnessChange -= 0.25;
        }

        _fitness += fitnessChange;
    }
}
```
