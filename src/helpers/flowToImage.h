#define FACTOR 1
#define PI 3.1415926536

// cartesianToRGB
void cartesianToRGB (float x, float y, float& R, float& G, float& B) {
  float radius = sqrt (x * x + y * y);
  if (radius > 1) radius = 1;
  float phi;
  if (x == 0.0)
    if (y >= 0.0) phi = 0.5 * PI;
    else phi = 1.5 * PI;
  else if (x > 0.0)
    if (y >= 0.0) phi = atan (y/x);
    else phi = 2.0 * PI + atan (y/x);
  else phi = PI + atan (y/x);
  float alpha, beta;    // weights for linear interpolation
  phi *= 0.5;
  // interpolation between red (0) and blue (0.25 * PI)
  if ((phi >= 0.0) && (phi < 0.125 * PI)) {
    beta  = phi / (0.125 * PI);
    alpha = 1.0 - beta;
    R = (int)(radius * (alpha * 255.0 + beta * 255.0));
    G = (int)(radius * (alpha *   0.0 + beta *   0.0));
    B = (int)(radius * (alpha *   0.0 + beta * 255.0));
  }
  if ((phi >= 0.125 * PI) && (phi < 0.25 * PI)) {
    beta  = (phi-0.125 * PI) / (0.125 * PI);
    alpha = 1.0 - beta;
    R = (int)(radius * (alpha * 255.0 + beta *  64.0));
    G = (int)(radius * (alpha *   0.0 + beta *  64.0));
    B = (int)(radius * (alpha * 255.0 + beta * 255.0));
  }
  // interpolation between blue (0.25 * PI) and green (0.5 * PI)
  if ((phi >= 0.25 * PI) && (phi < 0.375 * PI)) {
    beta  = (phi - 0.25 * PI) / (0.125 * PI);
    alpha = 1.0 - beta;
    R = (int)(radius * (alpha *  64.0 + beta *   0.0));
    G = (int)(radius * (alpha *  64.0 + beta * 255.0));
    B = (int)(radius * (alpha * 255.0 + beta * 255.0));
  }
  if ((phi >= 0.375 * PI) && (phi < 0.5 * PI)) {
    beta  = (phi - 0.375 * PI) / (0.125 * PI);
    alpha = 1.0 - beta;
    R = (int)(radius * (alpha *   0.0 + beta *   0.0));
    G = (int)(radius * (alpha * 255.0 + beta * 255.0));
    B = (int)(radius * (alpha * 255.0 + beta *   0.0));
  }
  // interpolation between green (0.5 * PI) and yellow (0.75 * PI)
  if ((phi >= 0.5 * PI) && (phi < 0.75 * PI)) {
    beta  = (phi - 0.5 * PI) / (0.25 * PI);
    alpha = 1.0 - beta;
    R = (int)(radius * (alpha * 0.0   + beta * 255.0));
    G = (int)(radius * (alpha * 255.0 + beta * 255.0));
    B = (int)(radius * (alpha * 0.0   + beta * 0.0));
  }
  // interpolation between yellow (0.75 * PI) and red (Pi)
  if ((phi >= 0.75 * PI) && (phi <= PI)) {
    beta  = (phi - 0.75 * PI) / (0.25 * PI);
    alpha = 1.0 - beta;
    R = (int)(radius * (alpha * 255.0 + beta * 255.0));
    G = (int)(radius * (alpha * 255.0 + beta *   0.0));
    B = (int)(radius * (alpha * 0.0   + beta *   0.0));
  }
  if (R<0) R=0;
  if (G<0) G=0;
  if (B<0) B=0;
  if (R>255) R=255;
  if (G>255) G=255;
  if (B>255) B=255;
}

void flowToImage(CTensor<float>& aFlow, CTensor<float>& aImage) {
  float aFactor = FACTOR*sqrt(0.5)*0.5;
  int aSize = aFlow.xSize()*aFlow.ySize();
  for (int i = 0; i < aSize; i++) {
    float R,G,B;
    cartesianToRGB(aFactor*aFlow.data()[i],aFactor*aFlow.data()[i+aSize],R,G,B);
    aImage.data()[i] = R;
    aImage.data()[i+aSize] = G;
    aImage.data()[i+2*aSize] = B;
  }
}