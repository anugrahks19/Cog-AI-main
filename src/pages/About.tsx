import { Shield, Users, Heart, Award } from "lucide-react";

const About = () => {
  return (
    <div className="min-h-screen">
      {/* Hero Section */}
      <section className="py-20 px-4 sm:px-6 lg:px-8 bg-gradient-brain">
        <div className="max-w-4xl mx-auto text-center">
          <h1 className="text-4xl sm:text-5xl font-bold text-foreground mb-6 animate-fade-in">
            About <span className="text-gradient">Cog.ai</span>
          </h1>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto animate-slide-up">
            We're on a mission to help families and doctors detect dementia early through advanced AI technology and accessible testing.
          </p>
        </div>
      </section>

      {/* Mission Section */}
      <section className="py-16 px-4 sm:px-6 lg:px-8">
        <div className="max-w-4xl mx-auto">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold text-foreground mb-4">Our Mission</h2>
            <p className="text-lg text-muted-foreground">
              Early detection saves lives and improves quality of life for millions of families worldwide.
            </p>
          </div>

          <div className="bg-card rounded-2xl p-8 shadow-soft">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8 items-center">
              <div>
                <h3 className="text-2xl font-semibold text-foreground mb-4">
                  Helping Families and Healthcare Providers
                </h3>
                <p className="text-muted-foreground mb-6">
                  Dementia affects over 55 million people worldwide, yet early detection remains challenging. Our AI-powered platform makes cognitive assessment accessible, accurate, and actionable.
                </p>
                <ul className="space-y-3">
                  <li className="flex items-center space-x-3">
                    <Shield className="h-5 w-5 text-primary" />
                    <span className="text-foreground">Privacy-first approach</span>
                  </li>
                  <li className="flex items-center space-x-3">
                    <Users className="h-5 w-5 text-primary" />
                    <span className="text-foreground">Family-centered care</span>
                  </li>
                  <li className="flex items-center space-x-3">
                    <Heart className="h-5 w-5 text-primary" />
                    <span className="text-foreground">Compassionate support</span>
                  </li>
                </ul>
              </div>
              <div className="bg-gradient-brain rounded-xl p-8 text-center">
                <div className="text-4xl font-bold text-primary mb-2">55M+</div>
                <p className="text-muted-foreground mb-4">People affected by dementia worldwide</p>
                <div className="text-4xl font-bold text-secondary mb-2">6M+</div>
                <p className="text-muted-foreground">Families seeking early detection</p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Story Section */}
      <section className="py-16 px-4 sm:px-6 lg:px-8 bg-muted/50">
        <div className="max-w-4xl mx-auto">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold text-foreground mb-4">Why Cog.ai Was Built</h2>
          </div>

          <div className="bg-card rounded-2xl p-8 shadow-soft">
            <div className="prose prose-lg max-w-none">
              <p className="text-muted-foreground leading-relaxed mb-6">
                Cog.ai was born from the recognition that early dementia detection could dramatically improve outcomes for millions of families. Current screening methods are often inaccessible, expensive, or require specialized medical facilities.
              </p>
              <p className="text-muted-foreground leading-relaxed mb-6">
                Our team of researchers, clinicians, and technologists came together with a shared vision: to democratize cognitive health screening through AI technology that's both scientifically rigorous and genuinely accessible.
              </p>
              <p className="text-muted-foreground leading-relaxed">
                By combining cutting-edge machine learning with human-centered design, we're making it possible for families to take proactive steps in cognitive health monitoring, while providing healthcare professionals with powerful tools for early intervention.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Values Section */}
      <section className="py-16 px-4 sm:px-6 lg:px-8">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold text-foreground mb-4">Our Values</h2>
            <p className="text-lg text-muted-foreground">
              The principles that guide everything we do
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {[
              {
                icon: Shield,
                title: "Privacy First",
                description: "Your cognitive health data is sensitive. We use state-of-the-art encryption and never share personal information without explicit consent."
              },
              {
                icon: Heart,
                title: "Compassionate Care",
                description: "We understand the emotional impact of cognitive health concerns. Our platform is designed with empathy and support at its core."
              },
              {
                icon: Award,
                title: "Scientific Rigor",
                description: "Every algorithm and assessment is backed by peer-reviewed research and validated through clinical studies."
              }
            ].map((value, index) => (
              <div key={index} className="bg-card rounded-2xl p-6 shadow-soft card-hover">
                <div className="p-3 bg-gradient-hero rounded-lg w-fit mb-4">
                  <value.icon className="h-6 w-6 text-white" />
                </div>
                <h3 className="text-xl font-semibold text-foreground mb-3">{value.title}</h3>
                <p className="text-muted-foreground">{value.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>
    </div>
  );
};

export default About;