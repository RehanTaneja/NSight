'use client';

export default function HomePage() {
  const features = [
    {
      title: "Generate text summaries instantly",
      icon: "✨",
      description: "Use the AI to generate text summaries instantly."
    },
    {
      title: "Analyze data and visualize graphs",
      icon: "📊",
      description: "Use the AI to analyze data and visualize graphs."
    },
    {
      title: "Answer questions using AI models",
      icon: "🤖",
      description: "Use the AI to answer questions using AI models."
    },
    {
      title: "Integrates with multiple platforms",
      icon: "🔗",
      description: "Use the AI to integrates with multiple platforms."
    }
  ];

  return (
    <div className="min-h-screen bg-linear-to-br from-blue-50 via-white to-purple-50">
      {/* Hero Section */}
      <div className="relative overflow-hidden bg-linear-to-r from-blue-600 to-purple-600 text-white">
        <div className="absolute inset-0 bg-black opacity-5"></div>
        <div className="relative max-w-6xl mx-auto px-8 py-20 text-center space-y-6">
          <h2 className="text-4xl font-bold text-gray-100 mb-10">
            NSight
          </h2>
          <p className="text-xl text-gray-100 max-w-2xl mx-auto font-medium">
            Welcome to your all-in-one AI tool. Generate, analyze, and visualize data quickly and effortlessly.
          </p>
        </div>
      </div>

      {/* Features Grid */}
      <section className="max-w-6xl mx-auto px-8 py-16">
        <div className="text-center mb-12">
          <h2 className="text-4xl font-bold text-gray-800 mb-4">
            Included Features
          </h2>
          <p className="text-gray-600 text-lg">
            Everything you need to supercharge your workflow
          </p>
        </div>
        
        {/* Grid container with border */}
        <div className="border-2 border-gray-200 rounded-3xl shadow-2xl bg-white/50 p-8 overflow-hidden">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            {features.map((feature, idx) => (
              <div
                key={idx}
                className="group relative bg-white rounded-2xl shadow-sm hover:shadow-2xl transition-all duration-300 overflow-hidden"
              >
                <div className="p-8 space-y-4 ">
                  <div className="flex flex-col items-center gap-4 text-center ">
                    <h3 className="text-xl font-bold text-gray-800 flex items-center ">
                      <div className="text-4xl bg-gradient-to-br from-blue-100 to-purple-100 p-2 rounded-xl">
                        {feature.icon}
                      </div>
                      &nbsp;&nbsp;{feature.title}
                    </h3>
                    <div className="text-center ">
                      <p className="text-gray-700 text-base leading-relaxed break-words">
                        {feature.description}
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>
    </div>
  );
}