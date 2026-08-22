abstract type AbstractLRScheduler end

# Helpers

get_initial_lr(s::AbstractLRScheduler) = error("get_initial_lr not implemented for $(typeof(s))")

step_scheduler!(s::AbstractLRScheduler,opt_state,epoch::Int,total_epochs::Int,loss;verbose::Bool=false) = error("step_scheduler! not implemented")

"""
  CosineAnnealing

A learning rate scheduler that implements the Cosine Annealing decay schedule.
It decreases the learning rate from a maximum value
to a minimum value following the shape of a half-cosine wave.

# Fields / Keyword Arguments
- `lr_max::Float32`: The initial, maximum learning rate (default: `0.001f0`).
- `lr_min::Float32`: The final, minimum learning rate (default: `1e-6f0`).
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
  ReduceLROnPlateau

Dynamic learning rate scheduler. Reduces the learning rate by a `factor`
when the loss has stopped improving for a given `patience` (number of epochs).
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