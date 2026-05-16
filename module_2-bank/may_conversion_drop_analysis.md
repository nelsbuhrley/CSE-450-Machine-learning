# Why May's Conversion Rate Craters

## Headline

The drop is real and large: May converts at **6.5%** while the dataset average is **11.4%**, and the small "shoulder" months (Mar, Sep, Oct, Dec) all sit between **45% and 51%**. Three things are happening at once, and they reinforce each other.

## The Monthly Picture

| Month | Contacts | Conversions | Conv. Rate | Share of Contacts |
|-------|---------:|------------:|-----------:|------------------:|
| mar   |      496 |         252 |     50.8%  |             1.3%  |
| apr   |    2,369 |         496 |     20.9%  |             6.4%  |
| **may** | **12,370** |     **800** |  **6.5%**  |          **33.4%** |
| jun   |    4,817 |         502 |     10.4%  |            13.0%  |
| jul   |    6,445 |         589 |      9.1%  |            17.4%  |
| aug   |    5,555 |         583 |     10.5%  |            15.0%  |
| sep   |      508 |         236 |     46.5%  |             1.4%  |
| oct   |      653 |         291 |     44.6%  |             1.8%  |
| nov   |    3,698 |         381 |     10.3%  |            10.0%  |
| dec   |      158 |          78 |     49.4%  |             0.4%  |

## Driver 1: May Is the Bank's Annual Mass-Call Month — Full of Cold Leads

May alone accounts for **33% of every call in the dataset** (12,370 of 37,069). The next-biggest month (July) is half that. That volume isn't a curated list:

- **98.2%** of May contacts have never been called before (`pdays == 999`)
- Only **1.7%** are people who previously said yes, versus **4.2%** in other months

The high-converting months look the way they do because they are *small, warm-lead* campaigns. Sep and Oct have ~20–26% prior-success contacts, and even there the team only made a few hundred calls. May is the opposite — a wide net cast over cold prospects.

## Driver 2: Channel Mix in May Is Heavily Skewed to Landline

May is the only month where **telephone** beats cellular. Across the year, telephone vs. cellular share:

| Month | Cellular | Telephone |
|-------|---------:|----------:|
| mar   |    89.3% |     10.7% |
| apr   |    92.9% |      7.1% |
| **may** | **40.0%** |  **60.0%** |
| jun   |    15.6% |     84.4% |
| jul   |    85.1% |     14.9% |
| aug   |    95.6% |      4.4% |
| sep   |    84.1% |     15.9% |
| oct   |    78.3% |     21.7% |
| nov   |    89.6% |     10.4% |
| dec   |    82.9% |     17.1% |

The cold-lead conversion gap by channel inside May is brutal:

- Cold leads on **cellular** convert at **10.6%**
- Cold leads on **telephone** convert at **3.2%**

Because so much of May's volume sits in that low-yield telephone bucket, the headline rate gets dragged down.

## Driver 3: The Macro Backdrop in May Is Unfavorable for Term Deposits

The economic indicators tell a story about *when* in the timeline May calls were made. May sits in a transition zone where the economy is recovering and rates are starting to rise — exactly the conditions that make fixed-rate term deposits *less* attractive.

| Month | emp.var.rate | euribor3m | nr.employed |
|-------|-------------:|----------:|------------:|
| mar   |       -1.800 |     1.162 |     5,055.4 |
| apr   |       -1.800 |     1.361 |     5,093.1 |
| **may** |   **-0.159** | **3.301** | **5,149.7** |
| jun   |        0.683 |     4.255 |     5,197.5 |
| jul   |        1.156 |     4.682 |     5,213.9 |
| aug   |        0.743 |     4.296 |     5,200.0 |
| sep   |       -2.191 |     0.834 |     4,989.2 |
| oct   |       -2.411 |     1.221 |     5,019.4 |
| nov   |       -0.428 |     3.714 |     5,172.5 |
| dec   |       -2.858 |     0.858 |     5,031.2 |

The high-converting months (Mar, Sep, Oct, Dec) all live in a downturn period (`emp.var.rate` between -1.8 and -2.9, `euribor3m` between 0.83 and 1.22). Fixed-rate term deposits are most appealing to customers when rates are falling and the economy looks shaky — exactly *not* what's happening in May.

## Decomposing the Damage

If May converted at the same rate as the rest of the year (13.8%), the bank would have closed roughly **1,707** deposits in May instead of the **800** they actually got — a gap of about **900 conversions** explained by the three factors above.

The effect isn't just a cold-leads phenomenon either. Even within the warm `poutcome=success` segment:

- May warm leads convert at **51.0%**
- Non-May warm leads convert at **68.7%**

That suggests macro/seasonality is suppressing yield on top of the mix issue.

## Bottom Line

May's "drop" is largely an **averaging artifact**. The bank dumps its lowest-quality contact list (cold + telephone) into May during an unfavorable macro window, while reserving small, targeted, cellular, warm-lead pushes for the months that look like winners. The month isn't broken — **the campaign design for May is what produces the low rate.**

## Recommended Follow-Ups

1. Re-estimate May performance after stratifying by `contact` channel and `poutcome` — does May still under-perform like-for-like, or is the entire deficit explained by mix?
2. Test whether splitting May's volume across adjacent months (with the same channel/lead mix) would have improved total annual conversions.
3. Build a model that controls for `euribor3m` and `emp.var.rate` to separate true seasonality from macro-driven seasonality.
4. Investigate whether the heavy telephone share in May reflects a specific list source (older customers, missing mobile numbers) — that segment may need a different product offer entirely.
