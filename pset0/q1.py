"""
Lucas Span-of-control (1978)

Consider a simplified version of Lucas (1978) span-ofcontrol model. Every
period, agents decide whether operate as an entrepreneur or work for the
market wage, w. In case the agent decides to be an entrepreneur, she runs a
business and hires workers to produce the final good using the following
production function:
    y = z * n^alpha    0 < alpha < 1
where z is the managerial ability and n is the number of workers hired.

Agents are heterogenous in their managerial ability, z, which has F(Z) as its
cdf, with support [z_min, inf).
"""
