import Base.assert, Base.showerror

type AssertionError<:Exception
  str::ASCIIString
end
function assert(cond::Bool, str::ASCIIString)
  if !cond;
    throw(AssertionError(str))
  end
end
Base.showerror(io::IO, e::AssertionError) = print(io, "assert: ", e.str);

function fft2{T<:FloatingPoint}(M::Array{T})
  return fft(M, (1,2));
end
function ifft2{T<:Complex}(M::Array{T})
  return ifft(M, (1,2));
end

function padarray{T}(A::Array{T, 3}, padsize::Tuple, padval::T, direction::ASCIIString)
  if direction!="both" && direction!="pre" && direction!="post";
    error("Invalid direction.")
  end

  m = size(A, 1); n = size(A, 2); d = size(A, 3);
  padm = padsize[1];
  padn = length(padsize)>1 ? padsize[2] : 0;
  padd = length(padsize)>2 ? padsize[3] : 0;

  newm = m; newn = n; newd = d;
  newi = 1; newj = 1; newk = 1;
  if direction=="pre" || direction=="both";
    newm += padm; newi += padm;
    newn += padn; newj += padn;
    newd += padd; newk += padd;
  end
  if direction=="post" || direction=="both";
    newm += padm;
    newn += padn;
    newd += padd;
  end

  newA = zeros(newm, newn, newd)+padval;
  newA[newi:newi+m-1, newj:newj+n-1, newk:newk+d-1] = A;
  return newA;
end

function shrink!{T<:FloatingPoint}(x::Array{T}, t::Array{T}, alpha::T)
  for i=1:length(t);
    x[i] = sign(t[i])*max(abs(t[i])-alpha, 0);
  end
  return x::Array{T};
end
function shrink{T<:FloatingPoint}(t::Array{T}, alpha::T)
  x = zeros(t);
  return shrink!(x, t, alpha);
end

function initcheck{T}(var::T, vartype::Type, varsize::Tuple, errmsg::ASCIIString)
  ret = var;
  if isempty(var);
    ret = zeros(vartype, varsize);
  else;
    assert(size(var)==varsize, errmsg)
  end
  return ret::T;
end

function outerprod!{T}(ZZt::SharedArray{T, 3}, Z::SharedArray{T, 2}, inds::UnitRange{Int})
  for i=inds;
    ZZt[:, :, i] = broadcast(*, Z[:, i], Z[:, i]');
  end
end

function outerprod_mul_D!{T}(ZZtd::SharedArray{T, 2}, Z::SharedArray{T, 2}, D::SharedArray{T, 2}, inds::UnitRange{Int})
  for i=inds;
    ZZtd[:, i] = broadcast(.*, Z[:, i], Z[:, i]'*D[:, i]);
  end
end
