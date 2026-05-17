---
layout: default
title: "Module 2 · Bank Marketing — Executive Summary"
project: module_2-bank
project_title: "Module 2 · Bank Marketing"
page_type: summary
permalink: /module_2-bank/page/summary/
---

{% include project-nav.html %}

<section class="exec-hero">
  <p class="exec-hero__eyebrow">Banco Federal de Finanças</p>
  <h1 class="exec-hero__title">Marketing Campaign Analysis</h1>
  <p class="exec-hero__subtitle">Prepared by North Wind Consulting</p>
  <div class="exec-hero__actions">
    <a class="btn-pill" href="{{ '/module_2-bank/page/assets/module_2-bank-executive-summary.pdf' | relative_url }}" download>
      ⬇  Download PDF
    </a>
    <a class="btn-pill btn-pill--ghost" href="{{ '/module_2-bank/page/technical/' | relative_url }}">
      Technical Deep-Dive →
    </a>
  </div>
</section>

## 1 · The Spray-and-Pray Tax

In the three years of your campaign, the month of May had the most calls — totalling **12,370 contacts** — and you lost **–$2.61 on each of them**. In May you did not hold back, reaching out to thousands of people in low-conversion groups that almost always lose money. By contrast, in your high-conversion months you called 1,815 people and earned **$15.58 per call**.

<div class="kpi-strip" role="list">
  <div class="kpi-strip__cell" role="listitem">
    <span class="kpi-strip__label">May mass calls</span>
    <span class="kpi-strip__value kpi-strip__value--loss">–$2.61</span>
  </div>
  <div class="kpi-strip__cell" role="listitem">
    <span class="kpi-strip__label">Campaign average</span>
    <span class="kpi-strip__value kpi-strip__value--loss">–$0.43</span>
  </div>
  <div class="kpi-strip__cell" role="listitem">
    <span class="kpi-strip__label">Targeted months</span>
    <span class="kpi-strip__value kpi-strip__value--gain">$15.58</span>
  </div>
</div>

![Value per call across May mass calls, campaign average, and targeted months]({{ '/module_2-bank/visualisation/output/marketing/white_bars/value_per_call_story.png' | relative_url }})

*Figure 1.  Value per call across May mass calls, campaign average, and targeted months.*
{: .figure-caption }

A targeted approach focused on high-value demographics is the answer; scaling that approach is the challenge. Our machine-learning (ML) model serves as the bridge between quantity and quality, allowing high-value campaigns to be scaled across tens of thousands of high-value prospects.

## 2 · What we can do for you

We built our model to help you reach consumers who are in need of your services — and to help you make money. In a recent trial campaign, you called 410 people and lost **$157**. Our model would have turned that $157 deficit into an **$824 profit**.

In your next proposed campaign, contacting all 4,100 people on your call list will likely lead to a loss of approximately **$1,600**. We project that once running your call list through our model, you will make **$7,300** by stripping your list down to the 480 highest-value prospects.

![Recent 410-contact test and the projected 4,119-contact campaign, with and without the model]({{ '/module_2-bank/visualisation/output/marketing/white_bars/campaign_value.png' | relative_url }})

*Figure 2.  Recent 410-contact test (left) and the projected 4,119-contact campaign (right), with and without the model.*
{: .figure-caption }

## 3 · People to call — and people not to

To maximize resource efficiency, our model cleanly separates the database into high-yield priority targets and immediate-skip candidates.

**The Golden List** focuses entirely on our high-conversion groups. At the very top are previously converted customers, who boast a phenomenal **65.8% conversion rate**. Recognizing this massive opportunity, the model shifts this group from a tiny 3% of our historical call list up to 21% of our total outreach. Similarly, students and retirees exhibit exceptionally strong conversion rates at 31.4% and 25.2% respectively. Retirees in particular represent high-liquidity accounts actively looking to lock in reliable term-deposit yields during fluctuating economic climates; the model scales our retiree outreach from a baseline of just 5% up to 23%.

**In stark contrast, the Black List** identifies the high-waste groups where our budget goes to die. Customers contacted via traditional landlines (5.3% conversion) and those in blue-collar employment blocks (6.8% conversion) drastically underperform our baseline. Historically, these two segments clogged our pipeline, consuming a staggering 58% of all marketing calls while returning almost no value. The model aggressively prunes this dead weight, shrinking their combined presence on the call list down to a lean 11% so your team stops wasting time on dead ends.

![Conversion rates by segment and the resulting call-list reshape]({{ '/module_2-bank/visualisation/output/marketing/white_bars/golden_blacklist.png' | relative_url }})

*Figure 3.  Conversion rates by segment (left) and the resulting call-list reshape (right).*
{: .figure-caption }

## 4 · Steps for improvement

The best metric we currently have to target the value of a customer is how likely they are to place money in a CD. To project the cost and value of each call we use the average time on a call (30 min) and the average amount of money placed in a CD ($4,700). Replacing those defaults with your internal labor and savings data is the single biggest lever for tightening the model's value estimates further.

---

**Want the full methodology?** The [Technical Deep-Dive]({{ '/module_2-bank/page/technical/' | relative_url }}) walks through the three candidate models, the cost-sensitive evaluation function, and the actual training code for each model.
