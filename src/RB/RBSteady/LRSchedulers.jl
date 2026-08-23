abstract type AbstractLRScheduler end

# Helpers

get_initial_lr(s::AbstractLRScheduler) = error("get_initial_lr not implemented for $(typeof(s))")

step_scheduler!(s::AbstractLRScheduler,opt_state,epoch::Int,total_epochs::Int,loss;verbose::Bool=false) = error("step_scheduler! not implemented")

"""
    mutable struct CosineAnnealing <: AbstractLRScheduler
      lr_max::Float32
      lr_min::Float32
    end

A learning rate scheduler that implements a Cosine Annealing decay schedule.
It smoothly decreases the learning rate from a maximum value down to a minimum value, following the shape of a half-cosine wave.

# Fields / Keyword Arguments
- `lr_max::Float32`: The initial, peak learning rate (default: `0.001f0`).
- `lr_min::Float32`: The final, minimum learning rate at the end of training (default: `1e-6f0`).
"""
mutable struct CosineAnnealing <: AbstractLRScheduler
  lr_max::Float32
  lr_min::Float32
end

CosineAnnealing(;lr_max=0.001f0,lr_min=1f-6) = CosineAnnealing(lr_max,lr_min)
  
get_initial_lr(s::CosineAnnealing) = s.lr_max

function step_scheduler!(scheduler::CosineAnnealing,opt_state,epoch::Int,total_epochs::Int,_loss;verbose::Bool=false)
  t = min(epoch,total_epochs)
  cos_val = cos(π * (t / total_epochs))
  new_lr = scheduler.lr_min + 0.5f0 * (scheduler.lr_max - scheduler.lr_min) * (1.0f0 + Float32(cos_val))
  
  Optimisers.adjust!(opt_state,new_lr)
end

"""
    mutable struct ReduceLROnPlateau <: AbstractLRScheduler
      patience::Int
      factor::Float32
      min_lr::Float32
      wait::Int
      best_loss::Float32
      current_lr::Float32
    end

A dynamic learning rate scheduler that reduces the learning rate by a multiplicative `factor` when the training loss has stopped improving for a specified number of epochs (`patience`).

# Keyword Arguments
- `patience::Int`: Number of epochs to wait without loss improvement before reducing the learning rate (default: `100`).
- `factor::Float32`: The multiplicative factor applied to the learning rate upon plateauing (default: `0.5f0`).
- `min_lr::Float32`: The absolute minimum learning rate boundary. The scheduler will not decay below this value (default: `1e-6f0`).
- `start_lr::Float32`: The initial learning rate at the beginning of the training (default: `0.001f0`).
"""
mutable struct ReduceLROnPlateau <: AbstractLRScheduler
  patience::Int
  factor::Float32
  min_lr::Float32
  wait::Int
  best_loss::Float32
  current_lr::Float32
end

ReduceLROnPlateau(;
  patience=100,
  factor=0.5f0,
  min_lr=1f-6,
  start_lr=0.001f0
) = ReduceLROnPlateau(patience,factor,min_lr,0,Inf32,start_lr)

get_initial_lr(s::ReduceLROnPlateau) = s.current_lr

function step_scheduler!(scheduler::ReduceLROnPlateau,opt_state,_epoch::Int,_total_epochs::Int,current_loss;verbose::Bool=false)
  if current_loss < scheduler.best_loss
    scheduler.best_loss = current_loss
    scheduler.wait = 0
  else
    scheduler.wait += 1
  end

  if scheduler.wait >= scheduler.patience
    new_lr = max(scheduler.current_lr * scheduler.factor,scheduler.min_lr)
    if new_lr < scheduler.current_lr
      verbose && @info "Plateau reached: LR decreased from $(scheduler.current_lr) to $new_lr"
      scheduler.current_lr = new_lr
      Optimisers.adjust!(opt_state,new_lr)
    end
    scheduler.wait = 0
  end
end