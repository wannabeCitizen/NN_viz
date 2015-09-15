//Michael Skirpan - Fast Forward Labs
//Visualization of simple Feed-forward and backpropagation algorithm
//Many thanks to:
//Daniel Shiffman - http://natureofcode.com - for education and example code
//P5.js folks - http://p5js.org - for example code
//Michael Nielson - http://neuralnetworksanddeeplearning.com/ - for lucid explanations of algorithms

var myNetwork;

function sigmoid(z){
    return 1.0/(1.0+Math.exp(-z));
}

function sigmoid_prime(z){
    return sigmoid(z)*(1-sigmoid(z));
}

// ############
// Neuron Object
// #############

var Neuron = function (location, bias, value) {
    //Set location, any initial value (for inputs), bias
    this.location = location || createVector(location[0], location[1]);
    this.connections = [];
    this.val = value || 0;
    this.bias = bias || random(-1, 1);
    this.acc = 0;
    this.incoming = 0;
    this.sig;

    //physical size
    this.r = 32;

    
    this.addConnection = function (c) {
        this.connections.push(c);
        c.b.incoming++;
    }  
  
  // Receive an input
  this.feedforward = function (input) {
    // Accumulate it
    this.val += input;
    this.acc++;

    // Activate it?
    if (this.acc == this.incoming) {
      this.val += this.bias;
      console.log()
      this.sig = sigmoid(this.val);
      this.acc = 0;
      if (this.sig > 0) {
        this.fire();
        }
      this.val = 0;
    } 
  }
  
  this.fire = function() {
    //make it bigger
    this.r = 64;   

    
    // We send the output through all connections
    for (c in this.connections) {
       this.connections[c].feedforward(this.val);
    } 
  }
  
  this.display = function() {
    stroke(0);
    strokeWeight(1);
    // Brightness is mapped to sum
    b = map(this.val,0,1,255,0);
    fill(b);
    ellipse(this.location.x, this.location.y, this.r, this.r);
    if (this.sig){
        push();
            fill(0, 102, 153);
            text(this.sig.toString(), this.location.x, this.location.y-32);
        pop();
    }

    //Will want to add numbers floating
    
    // Shrink interpolation
    this.r = lerp(this.r,32,0.1);
  }
}

// ###################
// Connection Object
// ###################

var Connection = function(a, b, weight) {
  // Connection is from Neuron A to B
  this.a = a;
  this.b = b;
  this.weight = weight || random(-1, 1);
  this.sending = false;
  this.output = 0;
  this.sender = createVector(this.a.x, this.a.y);
  
  // The Connection is active
  this.feedforward = function(val) {
    this.output = val*this.weight;        // Compute output
    this.sender.x = this.a.location.x;
    this.sender.y = this.a.location.y;
    this.sending = true;             // Turn on sending
  }
  
  // Update traveling sender
  this.update = function() {
    if (this.sending) {
      // Use a simple interpolation
      this.sender.x = lerp(this.sender.x, this.b.location.x, 0.1);
      this.sender.y = lerp(this.sender.y, this.b.location.y, 0.1);
      dist = p5.Vector.dist(this.sender, this.b.location);
      // If we've reached the end
      if (dist < 1) {
        // Pass along the output!
        b.feedforward(this.output);
        this.sending = false;
      }
    }
  }
  
  // Draw line and traveling circle
  this.display = function() {
    stroke(0);
    strokeWeight(1+(this.weight*4));
    line(this.a.location.x, this.a.location.y, this.b.location.x, this.b.location.y);

    if (this.sending) {
      fill(0);
      strokeWeight(1);
      ellipse(this.sender.x, this.sender.y, 16, 16);
    }
  }
}

var Network = function (inBoxes, neurons, connections, location) { 
  
  // The Network has a list of neurons
  this.state = 'forward'
  this.neurons = neurons || [];
  this.connections = connections || [];
  this.location = location || createVector(0, 0);
  this.learningRate = .01;
  this.inBoxes = inBoxes || [];

  // We can add a Neuron
  this.addNeuron = function (n) {
    this.neurons.push(n);
  }

  // We can connection two Neurons
  this.connect = function (a, b, weight) {
    c = new Connection(a, b, weight);
    a.addConnection(c);
    this.connections.push(c);
  } 
  
  // Sending an input to the first Neuron
  // We should do something better to track multiple inputs
  this.feedforward = function() {
    n1 = this.neurons[0];
    n1.fire();
    
    n2 = this.neurons[1];
    n2.fire();

    n3 = this.neurons[2];
    n3.fire();
    
  }
  
  // Update the animation
  this.update = function() {
    // Update backprop, eventually
    for (c in this.connections) {
      this.connections[c].update();
    }
  }
  
  // Draw everything
  this.display = function() {
    push();
        translate(this.location.x, this.location.y);
        for (n in this.neurons) {
          this.neurons[n].display();
        }

        for (c in this.connections) {
          this.connections[c].display();
        }
    pop();
  }
}

function add(a, b) {
    return a + b;
}

function setup() {
    var myCanvas = createCanvas(windowWidth, windowHeight);
    myCanvas.parent('processing');
    var layers = [3, 2, 1];
    //var numNeurons = layers.reduce(add);
    var neurons = [[], [], []];
    var forward = createButton('Forward');
    forward.position(windowWidth - 500, windowHeight - 50);
    forward.mousePressed(goForth);
    var backward = createButton('Backprop');
    backward.position(windowWidth - 800, windowHeight - 50);
    backward.mousePressed(backProp);


    //Setting up text boxes for the input layers
    var inLayers = [];
    for (i = 0; i<layers[0]; i++){
        input = createInput(random(-10, 10));
        inLayers[i] = input;
    }

    myNetwork = new Network(inLayers);


    //Create Neurons in order
    var currLayer = 0;
    for (i in layers){
        currLayer ++;
        for (var j = 0; j<layers[i]; j++){
            x = (currLayer * width) / 4;
            y = ((j+1)*height) / (layers[i] + 1);

            if (currLayer == 1) {
                input = inLayers[j];
                input.position(x-14, y-32);
                n = new Neuron(createVector(x,y), random(-1, 1), input.value());
                neurons[currLayer-1].push(n);
                myNetwork.addNeuron(n);

            } else { 
                n = new Neuron(createVector(x,y), random(-1, 1));
                neurons[currLayer-1].push(n);
                myNetwork.addNeuron(n);
            }
        }
    }

    // Create Connections
    for (var l = 0; l < layers.length; l++) {
        for (from in neurons[l]) {
            for (to in neurons[l+1]){
                myNetwork.connect(neurons[l][from], neurons[l+1][to], random(-1,1));
            }
        }
    }

}

function draw() {
    background(255);
    myNetwork.update();
    myNetwork.display();
}

function goForth() {
    newVals = [];
    for (i in myNetwork.inBoxes) {
        myNetwork.neurons[i].val = myNetwork.inBoxes[i].value();
    }
    myNetwork.feedforward();
}

function backProp() {
    background(255);
    myNetwork.backdate();
    myNetwork.display();
}

