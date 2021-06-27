import React, { useState } from "react";
import cn from "classnames";
import { graphs } from "./graphs";
import "./Insights.scss";

const Insights = () => {
  const [pick, setPick] = useState(graphs[0].id);
  const graph = graphs.find((g) => g.id === pick);
  const GraphCmp = graph.component;

  return (
    <div
      style={{
        position: "relative",
        width: "100%",
        height: "100%",
        overflow: "hidden",
      }}
    >
      <div
        style={{
          position: "absolute",
          zIndex: 1,
        }}
      >
        <div className="content-header white">Инсайты</div>
        <div className="graph-wrapper">
          <GraphCmp {...graph} />

          <div className="graph-selection">
            {graphs.map((graph) => (
              <div
                className={cn("pick", {
                  selected: graph.id === pick,
                })}
                style={{
                  fontSize: graph.fontSize,
                }}
                onClick={() => setPick(graph.id)}
              >
                {graph.pickTitle.map((word) => (
                  <div
                    className="pick-title"
                    style={{
                      lineHeight: graph.lineHeight,
                    }}
                  >
                    {word}
                  </div>
                ))}
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="square-bkg" />
    </div>
  );
};

export default Insights;
