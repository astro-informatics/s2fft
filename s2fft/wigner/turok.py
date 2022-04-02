from re import S
import numpy as np


def turok_quarter(dl: np.ndarray, beta: float, l: int) -> np.ndarray:
    """
    Added by Jason McEwen 19/11/05
    If beta=0 then dl=identity
    """
    if np.abs(beta) < 0:
        dl = np.identity(2 * l + 1, dtype=np.complex128)

    """
    Define a bunch of random numbers
    """
    # lp1 = l + 1
    lp1 = 1
    big_const = 1.0

    ang = beta
    c = np.cos(ang)
    s = np.sin(ang)
    si = 1.0 / s
    t = np.tan(-ang / 2.0)
    c2 = np.cos(ang / 2.0)
    omc = 1.0 - c

    """
    Define some vectors 
    """
    lrenorm = np.zeros(2 * l + 1, dtype=np.float64)
    cp = np.zeros(2 * l + 1, dtype=np.float64)
    cpi = np.zeros(2 * l + 1, dtype=np.float64)
    cp2 = np.zeros(2 * l + 1, dtype=np.float64)
    log_first_row = np.zeros(2 * l + 1, dtype=np.float64)
    sign = np.zeros(2 * l + 1, dtype=np.float64)

    """
    Compute first row.
    """
    log_first_row[0] = 2.0 * l * np.log(np.abs(c2))
    sign[0] = 1.0

    for i in range(2, 2 * l + 2):
        m = l + 1 - i
        ratio = np.sqrt((m + l + 1) / (l - m))
        log_first_row[i - 1] = log_first_row[i - 2] + np.log(ratio) + np.log(np.abs(t))
        sign[i - 1] = sign[i - 2] * t / np.abs(t)

    """
    Define some more random things
    """
    big = big_const
    lbig = np.log(big)
    bigi = 1.0 / big_const
    bigi2 = 1.0 / (big_const**2)
    xllp1 = l * (l + 1)

    """
    Initialising coefficients cp(m)= cplus(l-m).
    """

    for m in range(1, l + 1):
        xm = l - m
        cpi[m - 1] = 2.0 / np.sqrt(xllp1 - xm * (xm + 1))
        cp[m - 1] = 1.0 / cpi[m - 1]

    for m in range(2, l + 1):
        cp2[m - 1] = cpi[m - 1] * cp[m - 2]

    dl[1 - lp1, 1 - lp1] = 1.0
    dl[2 * l + 1 - lp1, 1 - lp1] = 1.0

    """
    Use recurrrence relation to fill columns down to diagonals.
    """

    for index in range(2, l + 2):
        dl[index - lp1, 1 - lp1] = 1.0
        lamb = ((l + 1) * omc - index + c) * si
        dl[index - lp1, 2 - lp1] = lamb * dl[index - lp1, 1 - lp1] * cpi[0]
        if index > 2:
            for m in range(2, index):
                lamb = ((l + 1) * omc - index + m * c) / s
                dl[index - lp1, m + 1 - lp1] = (
                    lamb * cpi[m - 1] * dl[index - lp1, m - lp1]
                    - cp2[m - 1] * dl[index - lp1, m - 1 - lp1]
                )
                if dl[index - lp1, m + 1 - lp1] > big:
                    lrenorm[index - 1] = lrenorm[index - 1] - lbig
                    for im in range(m + 1):
                        dl[index - lp1, im - lp1] = dl[index - lp1, im - lp1] * bigi

    """
    Other half of triangle.
    """

    for index in range(l + 2, 2 * l + 1):
        dl[index - lp1, 1 - lp1] = 1.0
        lamb = ((l + 1) * omc - index + c) / s
        dl[index - lp1, 2 - lp1] = lamb * dl[index - lp1, 1 - lp1] * cpi[0]
        if index < 2 * l:
            for m in range(2, 2 * l - index + 1):
                lamb = ((l + 1) * omc - index + m * c) / s
                dl[index - lp1, m + 1 - lp1] = (
                    lamb * cpi[m - 1] * dl[index - lp1, m - lp1]
                    - cp2[m - 1] * dl[index - lp1, m - 1 - lp1]
                )
                if dl[index - lp1, m + 1 - lp1] > big:
                    lrenorm[index - 1] = lrenorm[index - 1] - lbig
                    for im in range(1, m + 2):
                        dl[index - lp1, im - lp1] = dl[index - lp1, im - lp1] * bigi

    for i in range(1, l + 2):
        renorm = sign[i - 1] * np.exp(log_first_row[i - 1] - lrenorm[i - 1])
        for j in range(1, i + 1):
            dl[i - lp1, j - lp1] = dl[i - lp1, j - lp1] * renorm

    for i in range(l + 2, 2 * l + 2):
        renorm = sign[i - 1] * np.exp(log_first_row[i - 1] - lrenorm[i - 1])
        for j in range(1, 2 * l + 2 - i + 1):
            dl[i - lp1, j - lp1] = dl[i - lp1, j - lp1] * renorm

    return dl

    #   integer, intent(in) :: l
    #   real(kind = dp), intent(out) :: dl(-l:l,-l:l)
    #   real(kind = dp), intent(in) :: beta

    #   real(kind = dp) :: lambda, xllp1, xm, big, bigi, bigi2, c, s, omc, &
    #     lbig, expo, renorm, ratio, c2, t, si, ang, big_const
    #   real(kind = dp) :: cp(1:2 * l + 1)
    #   real(kind = dp) :: cpi(1:2 * l + 1)
    #   real(kind = dp) :: cp2(1:2 * l + 1)
    #   real(kind = dp) :: log_first_row(1:2 * l + 1)
    #   real(kind = dp) :: sign(1:2 * l + 1)
    #   real(kind = dp) :: lrenorm(1:2 * l + 1)

    #   real(kind = dp) :: ZERO_TOL = 1d-5
    #   if(abs(beta) < ZERO_TOL) then
    #     dl(-l:l,-l:l) = 0d0
    #     do i = -l,l
    #       dl(i,i) = 1d0
    #     end do
    #     return      # ** Exit routine
    #   end if

    #   lp1 = l + 1
    #   big_const = 1.0d150

    #   ang=beta
    #   c=dcos(ang)
    #   s=dsin(ang)
    #   si=1.d0/s
    #   t=dtan(-ang/2.d0)
    #   c2=dcos(ang/2.d0)
    #   omc=1.d0-c

    #   do i=1,2*l+1
    #     lrenorm(i)=0.d0
    #   end do

    #   """
    #   Compute first row.
    #   """

    #   log_first_row(1)=(2.d0*real(l,kind=dp))*dlog(dabs(c2))
    #   sign(1)=1.d0
    #   do i=2, 2*l+1
    #     m=l+1-i
    #     ratio=dsqrt(real(l+m+1,kind=dp)/real(l-m,kind=dp))
    #     log_first_row(i)=log_first_row(i-1) &
    #       +dlog(ratio)+dlog(dabs(t))
    #     sign(i)=sign(i-1)*t/dabs(t)
    #   end do

    #   big=big_const
    #   lbig=dlog(big)
    #   bigi=1.d0/big_const
    #   bigi2=1.d0/big_const**2
    #   xllp1=real(l*(l+1),kind=dp)

    #   """
    #   Initialising coefficients cp(m)= cplus(l-m).
    #   """

    #   do m=1,l
    #     xm=real(l-m,kind=dp)
    #     cpi(m)=2.d0/dsqrt(xllp1-xm*(xm+1))
    #     cp(m)=1.d0/cpi(m)
    #   end do
    #   do m=2,l
    #     cp2(m)=cpi(m)*cp(m-1)
    #   end do
    #   dl(1 - lp1, 1 - lp1)=1.d0
    #   dl(2*l+1 - lp1, 1 - lp1)=1.d0

    #   """
    #   Use recurrrence relation to fill columns down to diagonals.
    #   """
    #   do index= 2,l+1
    #     dl(index - lp1, 1 - lp1)=1.d0
    #     lambda=(real(l+1,kind=dp)*omc-real(index,kind=dp)+c)*si
    #     dl(index - lp1, 2 - lp1)=lambda*dl(index - lp1, 1 - lp1)*cpi(1)
    #     if(index.gt.2) then
    #       do m=2,index-1
    #         lambda=(real(l+1,kind=dp)*omc &
    #           -real(index,kind=dp)+real(m,kind=dp)*c)/s
    #         dl(index - lp1, m+1 - lp1)= &
    #           lambda*cpi(m)*dl(index - lp1, m - lp1)-cp2(m) &
    #           *dl(index - lp1, m-1 - lp1)
    #         if(dl(index - lp1, m+1 - lp1).gt.big) then
    #           lrenorm(index)=lrenorm(index)-lbig
    #           do im=1,m+1
    #             dl(index - lp1, im - lp1)=dl(index - lp1, im - lp1)*bigi
    #           end do
    #         end if
    #       end do
    #     end if
    #   end do

    #   """
    #   Other half of triangle.
    #   """

    #   do index= l+2,2*l
    #     dl(index - lp1, 1 - lp1)=1.d0
    #     lambda=(real(l+1,kind=dp)*omc-real(index,kind=dp)+c)/s
    #     dl(index - lp1, 2 - lp1)=lambda*dl(index - lp1, 1 - lp1)*cpi(1)
    #     if(index.lt.2*l) then
    #       do m=2,2*l-index+1
    #         lambda=(real(l+1,kind=dp)*omc-real(index,kind=dp) &
    #           +real(m,kind=dp)*c)/s
    #         dl(index - lp1, m+1 - lp1)= &
    #           lambda*cpi(m)*dl(index - lp1, m - lp1)-cp2(m)&
    #           *dl(index - lp1, m-1 - lp1)
    #         if(dl(index - lp1, m+1 - lp1).gt.big) then
    #           do im=1,m+1
    #             dl(index - lp1, im - lp1)=dl(index - lp1, im - lp1)*bigi
    #           end do
    #           lrenorm(index)=lrenorm(index)-lbig
    #         end if
    #       end do
    #     end if
    #   end do

    #   do i=1, l+1
    #     renorm=sign(i)*dexp(log_first_row(i)-lrenorm(i))
    #     do j=1, i
    #       dl(i - lp1, j - lp1)= dl(i - lp1, j - lp1)*renorm
    #     end do
    #   end do
    #   do i=l+2,2*l+1
    #     expo=log_first_row(i)-lrenorm(i)
    #     renorm=sign(i)*dexp(log_first_row(i)-lrenorm(i))
    #     do j=1,2*l+2-i
    #       dl(i - lp1, j - lp1)=dl(i - lp1, j - lp1)*renorm
    #     end do
    #   end do

    # end subroutine ssht_dl_beta_operator_quarter


def turok_fill(dl: np.ndarray, l: int) -> np.ndarray:

    # lp1 = l + 1
    lp1 = 1

    """
    Reflect across anti-diagonal
    """
    for i in range(1, l + 1):
        for j in range(l + 1, 2 * l + 1 - i + 1):
            dl[2 * l + 2 - i - lp1, 2 * l + 2 - j - lp1] = dl[j - lp1, i - lp1]

    """
    Reflect across diagonal
    """
    for i in range(1, l + 2):
        sgn = -1
        for j in range(i + 1, l + 2):
            dl[i - lp1, j - lp1] = dl[j - lp1, i - lp1] * sgn
            sgn = sgn * (-1)

    """
    Fill right matrix
    """
    for i in range(l + 2, 2 * l + 2):
        sgn = (-1) ** (i + 1)
        for j in range(1, 2 * l + 2 - i + 1):
            dl[j - lp1, i - lp1] = dl[i - lp1, j - lp1] * sgn
            sgn = sgn * (-1)

        for j in range(i, 2 * l + 2):
            dl[j - lp1, i - lp1] = dl[2 * l + 2 - i - lp1, 2 * l + 2 - j - lp1]

    for i in range(l + 2, 2 * l + 2):
        for j in range(2 * l + 3 - i, i - 1 + 1):
            dl[j - lp1, i - lp1] = dl[2 * l + 2 - i - lp1, 2 * l + 2 - j - lp1]

    return dl

    # subroutine ssht_dl_beta_operator_fill(dl, l)

    #   integer, intent(in) :: l
    #   real(kind = dp), intent(inout) :: dl(-l:l,-l:l)

    #   integer :: i, j,sgn, lp1

    #   lp1 = l + 1

    #   ! Reflect across anti-diagonal.

    #   do i = 1, l
    #     do j = l + 1, 2 * l + 1 - i
    #       dl(2 * l + 2 - i - lp1, 2 * l + 2 - j - lp1) = dl(j - lp1, i - lp1)
    #     end do
    #   end do

    #   ! Reflect across diagonal.

    #   do i = 1, l + 1
    #     sgn = - 1
    #     do j = i + 1, l + 1
    #       dl(i - lp1, j - lp1) = dl(j - lp1, i - lp1) * sgn
    #       sgn = sgn * (- 1)
    #     end do
    #   end do

    #   ! Fill in right quarter of matrix.

    #   do i = l + 2, 2 * l + 1
    #     sgn = (- 1)**(i + 1)
    #     do j = 1, 2 * l + 2 - i
    #       dl(j - lp1, i - lp1) = dl(i - lp1, j - lp1) * sgn
    #       sgn = sgn * (- 1)
    #     end do
    #     do j = i, 2 * l + 1
    #       dl(j - lp1, i - lp1) = dl(2 * l + 2 - i - lp1, 2 * l + 2 - j - lp1)
    #     end do
    #   end do

    #   do i = l + 2, 2 * l + 1
    #     do j = 2 * l + 3 - i, i - 1
    #       dl(j - lp1, i - lp1) = dl(2 * l + 2 - i - lp1, 2 * l + 2 - j - lp1)
    #     end do
    #   end do

    # end subroutine ssht_dl_beta_operator_fill
