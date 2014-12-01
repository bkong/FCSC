type FastCodingData{T}
  Dtx_fft::Array{Complex{T}, 3};
  DtD_fft::Array{Complex{T}, 3};
  convH::Int;
  convW::Int;
end
type FastCodingOpts{T}
  z::Array{T, 2};
  z_fft::Array{Complex{T}, 3};
  t::Array{T, 2};
  t_fft::Array{Complex{T}, 3};
  lambda_t::Array{T, 2};
  lambda_t_fft::Array{Complex{T}, 3};
  beta::T;       # 1
  mu_t::T;       # 1
  max_iter::Int; # 1000
  tol::T;        # 1e-10
  DEBUG::Bool;   # false
  debugData::Dict;
end
FastCodingOpts(T) = FastCodingOpts{T}(zeros(0, 0), zeros(0, 0, 0), zeros(0, 0), zeros(0, 0, 0), zeros(0, 0), zeros(0, 0, 0),
                      1, 1, 1000, 1e-10, false, Dict());


type LearningData{T}
  Ztx_fft::Array{Complex{T}, 2};
  ZtZ_fft::Array{Complex{T}, 3};
  convH::Int;
  convW::Int;
  filterSize::Int;
end
type LearningOpts{T}
  Dinit::Array{T, 3};
  s::Array{T, 3};
  lambda_s::Array{T, 3};
  mu_s::T;       # 1
  max_iter::Int; # 1000
  tol::T;        # 1e-10
  DEBUG::Bool;   # false
  debugData::Dict;
end
type LearningDebug{T}
  rerr::Array{T, 1};
  obj_d::Array{T, 1};
  obj_s::Array{T, 1};
  obj::Array{T, 1};
end
LearningOpts(T) = LearningOpts{T}(zeros(0, 0, 0), zeros(0, 0, 0), zeros(0, 0, 0), 1., 1000, 1e-10, false, Dict());


type FCSCOpts{T}
  Dinit::Array{T, 3};
  beta::T;       # 1
  mu_s::T;       # 1
  mu_t::T;       # 1
  max_iter::Int; # 1000
  tol::T;        # 1e-10
  DEBUG::Bool;   # false
  FIXED_D::Bool; # false
end
type FCSCDebug{T}
  norm_z::Array{T, 1};
  rerr::Array{T, 1};
  obj::Array{T, 1};
  dif_ds::Array{T, 1};
  dif_zt::Array{T, 1};
end
FCSCOpts(T) = FCSCOpts{T}(zeros(0, 0, 0), 1., 1., 1., 1000, 1e-10, false, false);
