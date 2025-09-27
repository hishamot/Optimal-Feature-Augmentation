<!DOCTYPE html>
<html>

<body>


<h2>Optimal Transformations Selection Problem</h2>
<p>This section focuses on the selection of those transformations that are statistically significant with positive correlation to the correct prediction of the model, referred to as optimal transformations in the remainder of this paper.</p>

<p>Let <span class="variable">F</span> denote the feature map at location <span class="variable">L</span> of the model and <span class="variable">t</span> denote a transformation set which has transformations like random vertical flip, random horizontal flip, random rotation, etc applied to obtain the transformed data. Then, the transformed feature map <span class="variable">F'</span> is defined as:</p>

<div class="equation">
F' = t(F) <span class="equation-number">(1)</span>
</div>

<p>Consider <span class="variable">n</span> arbitrary transformations <span class="variable">t₁, t₂, ..., tₙ</span> and the corresponding binary variables <span class="variable">x</span> = [x₁, x₂, ..., xₙ] which take value either 0 or 1. When <span class="variable">xᵢ</span> takes the value 1, then <span class="variable">tᵢ</span> is considered for performing the transformation on the feature maps. Let the input image (equivalent to the feature map at the input) be denoted as <span class="variable">F₀</span>. A logistic test [1] is used to identify the optimal transformations. For this, a binary response variable <span class="variable">y</span>,</p>

<div class="equation">
y = { 1; correct image classification<br>0; incorrect image classification } <span class="equation-number">(2)</span>
</div>

<p>and the probability of successfully classifying the image for a given <span class="variable">x</span>,</p>

<div class="equation">
p(x) = p(y=1/x) <span class="equation-number">(3)</span>
</div>

<p>are considered. As <span class="variable">x</span> can take 2ⁿ possible combinations of binary variables <span class="variable">xᵢ</span>, there will be a resultant probability value associated with each possible combination. As the value of <span class="variable">p(x)</span> is between 0 and 1 and <span class="variable">xᵢ</span>'s are independent binary variables, they are fitted using the logistic regression model defined by</p>

<div class="equation">
p(x) = exp(β₀ + Σβᵢxᵢ) / (1 + exp(β₀ + Σβᵢxᵢ)) <span class="equation-number">(4)</span>
</div>

<p>Here <span class="variable">β₀</span> to <span class="variable">βₖ</span> denote the coefficients of the model [1, 2]. Equation (4) can be rewritten in its logarithmic form as</p>

<div class="equation">
ln(p(x)/(1 - p(x))) = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ <span class="equation-number">(5)</span>
</div>

<p>The estimates of the coefficients <span class="variable">βₖ</span> can be obtained by the maximum likelihood method [3]. Subsequently, a hypothesis test is conducted on the regression coefficients to evaluate the significance of the associated transformation. The significance of each transformation is analyzed using the hypothesis test.</p>

<div class="equation">
H₀: βₖ = 0, H₁: βₖ ≠ 0 <span class="equation-number">(6)</span>
</div>

<p>where <span class="variable">H₀</span> represents the null hypothesis, and <span class="variable">H₁</span> represents the alternative hypothesis. If the null hypothesis (<span class="variable">H₀</span>) is accepted (p > α), where α denotes the significance level, the transformation <span class="variable">xᵢ</span>, associated with the coefficient <span class="variable">βᵢ</span>, is not statistically significant in the model. Conversely, if the alternative hypothesis (<span class="variable">H₁</span>) is accepted (p < α), the transformation <span class="variable">xᵢ</span> shows statistical significance [2]. The sign of the coefficient (<span class="variable">βᵢ</span>) indicates whether the corresponding transformation has a positive or negative correlation to the model performance. Finally, those transformations with statistically significant positive coefficients are selected. This selected subset of available transformations at the input of the model is the optimal transformations at <span class="variable">L₀</span>, denoted as <span class="variable">T₀</span>, <span class="variable">T₀</span> = { tᵢ ∈ T | xᵢ is significant with pᵢ < α and βᵢ > 0 }. The input feature map <span class="variable">F₀</span> is transformed using the selected optimal transformation <span class="variable">T₀</span>,</p>

<div class="equation">
F₀' = { T₀(F₀); if T₀ ≠ ∅<br>F₀; otherwise } <span class="equation-number">(7)</span>
</div>

<p>where ∅ denotes a null set. The transformed feature map <span class="variable">F₀'</span> is fed to the next layer, i.e., unlike normal data augmentation, only the transformed version of the input is fed to the next layer.</p>

<div class="figure">
<img src="Figure_1.png" alt="Architecture showing location-wise feature augmentation">
<div class="figure-caption">Figure 1: Architecture showing location-wise feature augmentation with the light green highlighted feature maps indicating those considered for optimal transformation selection, and the red highlighted feature maps correspond to transformed feature maps.</div>
</div>

<div class="subsection">
<h2>Optimal Transformations Selection at Lᵢ</h2>

<p>The optimal feature augmentation strategy can be divided into two-phase optimal feature selection phase and the actual model training phase.</p>

<p>Similar to finding optimal transformations at the input level (L₀), the approach can be extended to any intermediate level feature map Fᵢ at location Lᵢ. For this, first consider the features map F₁ (at L₁) defined as</p>

<div class="equation">
F₁ = C₁(F₀') <span class="equation-number">(8)</span>
</div>

<p>where C₁ denotes the convolution operation between L₀ and L₁. Similar to the input location L₀, at each intermediate location consider n arbitrary transformations t = t₁, t₂, ..., tₙ and the corresponding binary variables x = [x₁, x₂, ..., xₙ]. As discussed in the previous subsection, the coefficient values βᵢ and the corresponding p-value of the associated xᵢ in L₁ are computed. The resultant subset of optimal transformations at L₁ is denoted by T₁ and the corresponding transformed feature map is denoted by F'₁, where</p>

<div class="equation">
F₁' = T₁(F₁) <span class="equation-number">(9)</span>
</div>

<p>At the location L₁, augmented feature map F̃₁ is computed as</p>

<div class="equation">
F̃₁ = { [F₁, F₁']; if T₁ ≠ ∅<br>F₁; otherwise } <span class="equation-number">(10)</span>
</div>

<p>Using F̃₁ as the feature map at location L₁ and passing to the next layer of the model instead of F₁', in turn, doubles the computation for the rest of the model. Consequently, the feature map F₂ at location L₂ is given by</p>

<div class="equation">
F₂ = C₂(F̃₁) <span class="equation-number">(11)</span>
</div>

<p>where denoting C₂ as the convolution operation between locations L₁ and L₂. At location L₂, the selection of optimal features is performed using the feature map F₂ which has double the number of features compared to the feature map F₁. In the general case, the feature map at the location Lᵢ is given by</p>

<div class="equation">
Fᵢ = Cᵢ(F̃ᵢ₋₁) <span class="equation-number">(12)</span>
</div>

<p>where Cᵢ denotes the convolution block between locations Lᵢ₋₁ and Lᵢ and F̃ᵢ₋₁ is the augmented feature map at Lᵢ₋₁. With at least one transformation selected at each Lᵢ, Fᵢ has 2ⁱ⁻¹ times the number of feature maps as F₁. This Fᵢ is further used to find the set of optimal transformations Tᵢ at Lᵢ. Thus the augmented feature map at Lᵢ is given by</p>

<div class="equation">
Fᵢ' = Tᵢ(Fᵢ) <span class="equation-number">(13)</span>
</div>

<div class="equation">
F̃ᵢ = { [Fᵢ, Fᵢ'], if Tᵢ ≠ ∅<br>Fᵢ, otherwise } <span class="equation-number">(14)</span>
</div>
</div>


</body>
</html>
