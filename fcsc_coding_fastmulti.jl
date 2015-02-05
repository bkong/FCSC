# Helper functions
function z_subproblem!{T}(z_fft::Array{Complex{T}, 3},
                          t_fft::Array{Complex{T}, 3}, mu_t::T, lambda_t_fft::Array{Complex{T}, 3},
                          DtD_fft::Array{Complex{T}, 3}, Dtx_fft::Array{Complex{T}, 3})
  convLen = size(t_fft, 3);
  dictSize = size(DtD_fft, 1);
  numImgs = size(Dtx_fft, 2);
  invmu_t = 1/mu_t;
  # (Eq. 7)
  invA = zeros(Complex128, dictSize, dictSize); b = zeros(Complex128, dictSize, numImgs);
  for i=1:convLen;
    # z_fft[:, :, i] = ( DtD_fft[:, :, i] + mu_t.*I ) \
    #   ( Dtx_fft[:, :, i] + mu_t.*t_fft[:, :, i] - lambda_t_fft[:, :, i] )
    # using the Sherman-Morrison formula
    a = 1 + invmu_t*sum(diag(DtD_fft[:, :, i]));
    scalar = -1/(a*mu_t*mu_t);
    broadcast!(*, invA, scalar, DtD_fft[:, :, i]);
    for j=1:dictSize; invA[j, j] = invmu_t + invA[j, j]; end
    copy!(b, Dtx_fft[:, :, i]);
    b += mu_t.*t_fft[:, :, i];
    b -= lambda_t_fft[:, :, i];
    z_fft[:, :, i] = invA*b;
  end
  return z_fft::Array{Complex{T}, 3};
end

function t_subproblem!{T}(t::Array{T, 2}, z::Array{T, 2}, mu_t::T, lambda_t::Array{T, 2}, beta::T)
  # (Eq. 9)
  if mu_t>0;
    # shrink!(t, z+lambda_t/mu_t, beta/mu_t);
    for i=1:length(t);
      t[i] = lambda_t[i]/mu_t + z[i];
    end
    shrink!(t, t, beta/mu_t);
  else;
    fill!(t, 0);
  end
  return t::Array{T, 2};
end


function fcsc_coding_fastmulti{T}(data::FastCodingData{T}, optParams::FastCodingOpts{T})
# Alg. parameters:
#   mu_max: 1e5
#   tau: [1.01 1.5] larger means faster convergence but sacrifices primal feasibility,
#        smaller means slower convergence

  const mu_max = 1e5;
  const tau = 1.01;

  const Dtx_fft      = data.Dtx_fft;
  const DtD_fft      = data.DtD_fft;
  const convH        = data.convH;
  const convW        = data.convW;
        z            = optParams.z;
        z_fft        = optParams.z_fft;
        t            = optParams.t;
        t_fft        = optParams.t_fft;
        lambda_t     = optParams.lambda_t;
        lambda_t_fft = optParams.lambda_t_fft;
  const beta::T      = optParams.beta;
        mu_t::T      = optParams.mu_t;
  const max_iter     = optParams.max_iter;
  const tol          = optParams.tol;
  const DEBUG        = optParams.DEBUG;
        debugData    = optParams.debugData;



  numImgs  = size(Dtx_fft, 2);
  dictSize = size(DtD_fft, 1);
  convLen  = convH*convW;

  assert(size(DtD_fft, 3)==convLen, "DtD_fft does not have the correct depth.")
  assert(size(Dtx_fft, 1)==dictSize && size(Dtx_fft, 3)==convLen,
    "Dtx_fft does not have the correct dimensions.")
  z = initcheck(z, T, (convLen*dictSize, numImgs), "z does not have the correct dimensions.");
  z_fft = initcheck(z_fft, Complex{T}, (dictSize, numImgs, convLen), "z_fft does not have the correct dimensions.");
  update_t = false;
  if vecnorm(z)==0;
    # Initialize z to pure reconstruction solution
    fill!(z_fft, 0);
    # Increase from 1e-14 if singular errors still occur.
    I = 1e-14.*eye(dictSize);
    for i=1:convLen;
      z_fft[:, :, i] = ( DtD_fft[:, :, i]+I ) \ Dtx_fft[:, :, i];
    end
    for j=1:numImgs;
      z_tmp = reshape(squeeze(z_fft[:, j, :], 2).', convH, convW, dictSize);
      z[:, j] = reshape(real(ifft2(z_tmp)), dictSize*convLen, 1);
    end
    update_t = true;
  end

  t = initcheck(t, T, (convLen*dictSize, numImgs), "t does not have the correct dimensions.");
  lambda_t = initcheck(lambda_t, T, (convLen*dictSize, numImgs), "lambda_t does not have the correct dimensions.");
  lambda_t_fft = initcheck(lambda_t_fft, Complex{T}, (dictSize, numImgs, convLen), "lambda_t_fft does not have the correct dimensions.");
  if isempty(t_fft);
    t_fft = zeros(Complex{T}, dictSize, numImgs, convLen);
    update_t = true;
  else;
    assert(size(t_fft)==(dictSize, numImgs, convLen), "t_fft does not have the correct dimensions.")
  end
  if update_t;
    copy!(t, z);
    copy!(t_fft, z_fft);
  end



  if DEBUG;
    debugData["z"] = z;
    debugData["t"] = t;
    debugData["lambda_t"] = lambda_t;
    (prev_obj, rerr, spar) = compute_lagrangian(debugData);
  end



  iter = 0;
  converged = false;
  while ~converged;
    iter += 1;

    if max_iter>1; @printf("\b\b\b%03d", iter); end

    # Solve for z* in the frequency domain
    z_subproblem!(z_fft, t_fft, mu_t, lambda_t_fft, DtD_fft, Dtx_fft);
    # TODO parfor this loop
    for j=1:numImgs;
      z_tmp = reshape(squeeze(z_fft[:, j, :], 2).', convH, convW, dictSize);
      z[:, j] = reshape(real(ifft2(z_tmp)), dictSize*convLen, 1);
    end
    if DEBUG;
      debugData["z"] = z;
      (obj, rerr, spar) = compute_lagrangian(debugData);
      is_mono = (prev_obj>=obj) ? 'O' : 'X';
      @printf("          Z: %c obj=%.10f (rerr=%.10f, spar=%.10f)\n", is_mono, obj, rerr, spar);
      prev_obj = obj;
    end


    # Solve for t* in the spatial domain
    t_subproblem!(t, z, mu_t, lambda_t, beta);
    if DEBUG;
      debugData["t"] = t;
      (obj, rerr, spar) = compute_lagrangian(debugData);
      is_mono = (prev_obj>=obj) ? 'O' : 'X';
      @printf("          T: %c obj=%.10f (rerr=%.10f, spar=%.10f)\n", is_mono, obj, rerr, spar);
      prev_obj = obj;
    end


    if max_iter==1; return (z, t); end # Don't do unnecessary work
    # TODO parfor this loop
    for j=1:numImgs;
      t_fft[:, j, :] = reshape(fft2(reshape(t[:, j], convH, convW, dictSize)), convLen, dictSize).';

      # Update the Lagrangian multipliers
      for i=1:convLen*dictSize; lambda_t[i, j] += mu_t*( z[i, j]-t[i, j] ); end
      lambda_t_fft[:, j, :] = reshape(fft2(reshape(lambda_t[:, j], convH, convW, dictSize)), convLen, dictSize).';
    end
    if DEBUG;
      debugData["lambda_t"] = lambda_t;
      (obj, rerr, spar) = compute_lagrangian(debugData);
      is_mono = (prev_obj>=obj) ? 'O' : 'X';
      @printf("          L: %c obj=%.10f (rerr=%.10f, spar=%.10f)\n", is_mono, obj, rerr, spar);
      prev_obj = obj;
    end

    # Update penalty term
    mu_t = min(tau*mu_t, mu_max);
    if DEBUG;
      debugData["mu_t"] = mu_t;
      (obj, rerr, spar) = compute_lagrangian(debugData);
      is_mono = (prev_obj>=obj) ? 'O' : 'X';
      @printf("          M: %c obj=%.10f (rerr=%.10f, spar=%.10f)\n", is_mono, obj, rerr, spar);
      prev_obj = obj;
    end



    codesConverged = true;
    for j=1:numImgs;
      if mu_t>0 && mean((z[:, j]-t[:, j]).^2)>tol;
        codesConverged = false;
        break;
      end
    end
    if iter>=max_iter || codesConverged;
      converged = true;
    end
  end

  return (z, t);
end
