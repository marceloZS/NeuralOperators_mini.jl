using FFTW
using NPZ
using Plots
using NeuralOperators
using Lux, Zygote, Optimisers, Random
using Statistics

train_file = "./TESIS/NEURAL_OPS/train_for_julia.npz"
test_file  = "./TESIS/NEURAL_OPS/test_for_julia.npz"

data_train = npzread(train_file)
x_train = convert(Array{Float32}, data_train["x"])
y_train = convert(Array{Float32}, data_train["y"])
@info "Loaded train: x_train size=$(size(x_train)), y_train size=$(size(y_train))"

data_test = npzread(test_file)
x_test = convert(Array{Float32}, data_test["x"])
y_test = convert(Array{Float32}, data_test["y"])
@info "Loaded test: x_test size=$(size(x_test)), y_test size=$(size(y_test))"

# Nueva interpretación: (grid_size, channels, batch)
(N_grid, Chan_in, B_train)  = size(x_train)
(_,       Chan_out, _)        = size(y_train)
println("Train: grid size = $N_grid, channels_in = $Chan_in, batch_train = $B_train, channels_out = $Chan_out")

(N_grid2, Chan_in2, B_test)  = size(x_test)
(_,       Chan_out2, _)       = size(y_test)
println("Test: grid size = $N_grid2, channels_in = $Chan_in2, batch_test = $B_test, channels_out = $Chan_out2")

@assert Chan_in == Chan_in2 "Input channels mismatch!"
@assert Chan_out == Chan_out2 "Output channels mismatch!"

rng = Random.default_rng()
model = FourierNeuralOperator(Lux.gelu; chs=(Chan_in, 32, 32, 32, Chan_out), modes=(12,))
global ps, st = Lux.setup(rng, model)

function eachbatch(x::Array{Float32,3}, y::Array{Float32,3}; bs::Int)
    n = size(x, 3)  # batch dimension es la tercera
    batches = Vector{Tuple{Array{Float32,3},Array{Float32,3}}}()
    i = 1
    while i <= n
        j = min(i + bs - 1, n)
        bx = x[:, :, i:j]  # (grid, channels, batch_slice)
        by = y[:, :, i:j]
        push!(batches, (bx, by))
        i = j + 1
    end
    return batches
end

mse(a,b) = mean((a .- b).^2)
function loss(ps, st, bx, by)
    ŷ, st2 = model(bx, ps, st)
    return mse(ŷ, by), st2
end

function train_loop!(model, ps, st, opt, x_train, y_train, n_epochs, batch_size)
    for epoch in 1:n_epochs
        lsum = 0.0
        nb   = 0
        for (bx, by) in eachbatch(x_train, y_train; bs=batch_size)
            (l, st_new), back = Zygote.pullback((ps_inner) -> loss(ps_inner, st, bx, by), ps)
            gs = back((one(l), nothing))[1]
            ps = Optimisers.update!(opt, ps, gs)
            st = st_new
            lsum += l
            nb   += 1
        end
        println("Epoch $epoch | avg MSE = $(lsum/nb)")
    end
    return ps, st
end

const BATCH_SIZE = 16
const N_EPOCHS = 5

opt = Optimisers.setup(Optimisers.Adam(1e-3), ps)

println("Training for $N_EPOCHS epochs...")
ps, st = train_loop!(model, ps, st, opt, x_train, y_train, N_EPOCHS, BATCH_SIZE)

println("Evaluating on test set…")
lsum = 0.0
nb   = 0
for (bx, by) in eachbatch(x_test, y_test; bs=BATCH_SIZE)
    ŷ, _ = model(bx, ps, st)
    lsum += mse(ŷ, by)
    nb   += 1
end
test_mse = lsum/nb
println("Test MSE = ", test_mse)

i = 1
x1 = x_test[:, :, i:i] 
y1 = y_test[:, :, i:i]
ŷ1, _ = model(x1, ps, st)
plot(1:N_grid, y1[:, 1, 1], label="y true", color=:blue)
plot!(1:N_grid, ŷ1[:, 1, 1], label="ŷ pred", linestyle=:dash, color=:red)
xlabel!("Grid index")
ylabel!("Value")
title!("True vs Predicted – Test sample $i")
savefig("true_vs_pred_test_sample1.png")
println("Saved plot: true_vs_pred_test_sample1.png")

println("Finished.")